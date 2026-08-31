# IVF

![IVF: Voronoi partition over k-means centroids; only the scan_buckets_count buckets closest to the query are scanned, with an optional precise rerank](../figures/indexes/ivf-overview.svg)

IVF (Inverted File) is VSAG's **partition-based** index. It clusters the corpus into
buckets at build time, and at query time only scans the buckets whose centroids are
closest to the query. This turns an O(N) linear scan into O(N · `scan_buckets_count`
/ `buckets_count`) with tunable recall/latency.

IVF trades a little recall (compared to graph indexes) for lower memory overhead,
higher throughput on batch workloads, and simpler sharding — which makes it a good
default when the corpus is large (hundreds of millions or more), when memory is
tight, or when queries are naturally parallelizable.

- Source: `src/algorithm/ivf.{h,cpp}`, `src/algorithm/ivf_parameter.{h,cpp}`
- Example: [`examples/cpp/106_index_ivf.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/106_index_ivf.cpp)

## How it works

1. **Clustering.** A sample of the dataset is clustered with k-means (or sampled
   randomly, `ivf_train_type: "random"`) to produce `buckets_count` centroids.
2. **Assignment.** Every vector is written to the inverted list of its nearest
   centroid, stored in the configured coarse quantization (`base_quantization_type`).
   Optionally, a second high-precision copy is kept (`use_reorder: true`) for
   post-filter reordering.
3. **Search.** For each query, the `scan_buckets_count` nearest centroids are
   computed first, then the vectors in those buckets are scored. When reordering is
   enabled, `factor` controls how many extra candidates are fetched from the coarse
   stage before being re-scored with the precise quantizer.

A second partition strategy, **GNO-IMI** (`partition_strategy_type: "gno_imi"`),
splits the space into two orthogonal sets of centroids
(`first_order_buckets_count` × `second_order_buckets_count`) for even finer
partitioning on very large corpora.

## Quick start

```cpp
#include <vsag/vsag.h>

std::string params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "buckets_count": 256,
        "base_quantization_type": "sq8",
        "partition_strategy_type": "ivf",
        "ivf_train_type": "kmeans"
    }
})";
auto index = vsag::Factory::CreateIndex("ivf", params).value();

// Build.
auto base = vsag::Dataset::Make();
base->NumElements(n)->Dim(128)->Ids(ids)->Float32Vectors(data)->Owner(false);
index->Build(base);

// Search.
auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(128)->Float32Vectors(q)->Owner(false);
auto result = index->KnnSearch(
    query, /*topk=*/10,
    R"({"ivf": {"scan_buckets_count": 16}})").value();
```

## Build parameters

Build-time parameters live under `index_param`. See
[Index Parameters](../resources/index_parameters.md) for the exhaustive list.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `partition_strategy_type` | string | `"ivf"` | `ivf` (single-level) or `gno_imi` (two-level orthogonal) |
| `buckets_count` | int | `10` | Number of inverted lists (effective for `ivf`) |
| `first_order_buckets_count` | int | `10` | First-level count (effective for `gno_imi`) |
| `second_order_buckets_count` | int | `10` | Second-level count (effective for `gno_imi`) |
| `ivf_train_type` | string | `"kmeans"` | Centroid training: `kmeans` or `random` |
| `enable_gpu_build` | bool | `false` | Allow the CUDA backend while training centroids; see [GPU-accelerated training](#gpu-accelerated-training) |
| `gpu_device_id` | int | `0` | CUDA device ordinal; an ordinal the machine does not have keeps the build on the CPU |
| `gpu_memory_budget` | int | `0` | Ceiling in bytes on the device working set; `0` derives it from what the device has free |
| `gpu_min_work_threshold` | int | `0` | Smallest `count * buckets_count * dim` worth offloading; `0` uses the calibrated default |
| `route_max_degree` | int | `64` | Routing HGraph maximum degree (effective for `ivf`) |
| `route_ef_construction` | int | `300` | Routing HGraph construction search breadth (effective for `ivf`) |
| `base_quantization_type` | string | `"fp32"` | `fp32`, `fp16`, `bf16`, `sq8`, `sq4`, `sq8_uniform`, `sq4_uniform`, `pq`, `pqfs`, `rabitq` — see the [Quantization chapter](../quantization/README.md) for per-quantizer details |
| `base_pq_dim` | int | `1` | PQ subspaces (required with `pq` / `pqfs`) |
| `rabitq_pca_dim` | int | `0` | Optional PCA preprocessing dimension for `base_quantization_type: "rabitq"` |
| `rabitq_bits_per_dim_query` | int | `32` | Query bits for `rabitq`; allowed values are `4` or `32` |
| `rabitq_bits_per_dim_base` | int | `1` | Stored-code bits for `rabitq`; allowed range is `[1, 8]` |
| `rabitq_version` | string | `"standard"` | `rabitq` layout: `"standard"` or `"split_1bit_7bit"` |
| `rabitq_error_rate` | float | `1.9` | Positive error-budget parameter for `rabitq` encoding |
| `rabitq_use_fht` | bool | `false` | Enable FHT rotation before `rabitq` binarization |
| `fast_encode_rabitq` | bool | `true` | Use CAQ fast construction for multi-bit `rabitq`; set to `false` for exact encoding |
| `fast_encode_rabitq_rounds` | int | `6` | CAQ adjustment rounds; allowed range is `[1, 32]` |
| `use_reorder` | bool | `false` | Keep a high-precision copy and re-rank after the coarse scan |
| `precise_quantization_type` | string | `"fp32"` | Quantizer used for reordering (with `use_reorder: true`) |
| `precise_codes_layout` | string | `"flat"` | Storage layout for precise codes: `"flat"` keeps the legacy one-code-per-vector layout; `"bucket"` stores the precise code in the same bucket and offset as its basic posting |
| `base_io_type` | string | `"memory_io"` | Storage backend for coarse codes; supports `uring_io` when built with liburing |
| `precise_io_type` | string | `"block_memory_io"` | Storage backend for precise codes (`memory_io`, `block_memory_io`, `mmap_io`, `buffer_io`, `async_io`, `uring_io`, `reader_io`) |
| `precise_file_path` | string | `""` | File path when the precise IO type is disk-backed |

`precise_codes_layout: "bucket"` requires `use_reorder: true`. It supports
`memory_io`, `block_memory_io`, `buffer_io`, `async_io`, and `uring_io`
(when io_uring is available).
`mmap_io` and `pqfs` precise quantization are not supported. The bucket layout currently requires
`buckets_per_data: 1`; configurations that assign one vector to multiple buckets are rejected.

For both `flat` and `bucket` layouts, serialized precise codes can be loaded read-only from an
external `Reader`. Pass `precise_io_type: "reader_io"` and `precise_reader` to `Index::Load`.
The reader must expose exactly the `high_precision_codes` block payload for `flat`, or the
`ivf_precise_bucket` block payload for `bucket`. `reader_io` uses the normal read cache when
`precise_enable_read_cache` is enabled.

```cpp
vsag::LoadParameters load_parameters;
load_parameters.Set("precise_io_type", "reader_io")
    .Set("precise_enable_read_cache", true)
    .Set("precise_cache_total_size", 256ULL * 1024 * 1024)
    .SetReader("precise_reader", precise_codes_reader);
auto loaded = vsag::Index::Load(stream, load_parameters).value();
```

`reader_io` is a load-and-query placement policy. Build the index with a writable precise IO,
serialize it, and then use the external reader when loading it for search.

A rule of thumb for `buckets_count` is `sqrt(N)` to `4 * sqrt(N)` where `N` is the
corpus size.

## GPU-accelerated training

Clustering dominates an IVF build once `buckets_count` grows, because the training
sample scales with it: `train_sample_count` defaults to
`max(65536, 64 * buckets_count)`. With `ivf_train_type: "kmeans"` the seeding pass
alone is `O(buckets_count * train_sample_count * dim)` and runs single-threaded.

When VSAG is built with `ENABLE_CUDA=ON`, that work can run on a CUDA device
instead. Set `enable_gpu_build` to turn it on:

```json
{
    "buckets_count": 4096,
    "base_quantization_type": "fp32",
    "partition_strategy_type": "ivf",
    "ivf_train_type": "kmeans",
    "enable_gpu_build": true,
    "gpu_device_id": 0
}
```

Three parts of training move to the device: k-means++ seeding, nearest-centroid
assignment, and centroid accumulation. Only `ivf_train_type: "kmeans"` benefits;
`random` does not cluster and is unaffected.

**The index format is unchanged.** An index trained on a device serializes to the
same format as one trained on the CPU, and can be loaded and searched on a machine
with no GPU. The centroid values themselves differ, as they already do between two
CPU runs, because seeding starts from a random draw. Assignment agrees with the CPU
path except where two centroids are equidistant from a point to within
floating-point rounding, so recall matches to within run-to-run noise.

**The build falls back to the CPU, silently and with identical results**, when VSAG
was built without CUDA, no device is present, `gpu_device_id` names an ordinal the
machine does not have, the problem is below `gpu_min_work_threshold`, the device
cannot hold what a step needs, or any CUDA call fails. A build never fails because
of the backend.

**Memory.** Points are streamed in chunks, so the corpus does not have to fit in
device memory; only the centroids and one chunk do. The working set is bounded by
`gpu_memory_budget`, which by default is derived from what the device reports free
so the same build behaves on a small card and makes use of a large one. Seeding is
the exception: it sweeps the whole training sample once per centroid, so the sample
itself has to be resident, and a sample too large for the device sends seeding back
to the CPU while the other two parts stay on it.

Measured on SIFT1M (1M base vectors, 128 dimensions, 48 threads, one RTX 3090
against two Xeon Gold 6336Y):

| `buckets_count` | CPU build | GPU build | Speedup | Recall@10 change |
|-----------------|-----------|-----------|---------|------------------|
| 1000 | 59.9 s | 55.9 s | 1.07x | −0.02 pt |
| 4096 | 180.1 s | 61.6 s | 2.93x | +0.03 pt |
| 16384 | 781.5 s | 85.6 s | 9.13x | −0.06 pt |

The speedup grows with `buckets_count` because clustering takes a larger share of
the build. At small `buckets_count` most of the time goes to assigning vectors to
buckets, which stays on the CPU.

## Search parameters

Search-time parameters live under the `ivf` sub-object:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scan_buckets_count` | int | — (required) | Number of buckets probed per query. Must be ≤ `buckets_count` (except when `disable_bucket_scan` is true, where larger values are allowed and unavailable slots are padded with `-1`). |
| `disable_bucket_scan` | bool | `false` | Return bucket IDs and distances. Supports batch queries. |
| `factor` | float | `2.0` | With reordering enabled, pulls `factor * topk` coarse candidates before the precise rescore. |
| `enable_reorder` | bool | `true` | Set to `false` to skip the final reorder stage for this request even when the index was built with reorder enabled. |
| `parallelism` | int | `1` | Threads used to scan buckets in parallel for a single query. |
| `timeout_ms` | double | `+∞` | Hard cap in milliseconds; partial results are returned once exceeded. |

```cpp
auto result = index->KnnSearch(
    query, topk,
    R"({"ivf": {"scan_buckets_count": 32, "factor": 2.0, "parallelism": 4}})").value();
```

```cpp
auto fast_result = index->KnnSearch(
    query, topk,
    R"({"ivf": {"scan_buckets_count": 32, "factor": 2.0, "enable_reorder": false}})").value();
```

## When to use IVF

- Large corpora (hundreds of millions of vectors and above), especially when the
  working set does not fit comfortably in RAM.
- Batch or high-throughput workloads where per-query latency is less critical than
  queries-per-second.
- Memory-tight deployments that benefit from aggressive quantization (`sq8`,
  `sq4_uniform`, `pq`, `pqfs`) combined with `use_reorder` to recover recall.
- Shard-friendly setups: buckets map naturally onto shards or disk blocks.

For latency-sensitive, high-recall workloads on dense embeddings, compare against
[HGraph](hgraph.md) first.

## See also

- [Creating an Index](../guide/create_index.md)
- [Index Parameters](../resources/index_parameters.md)
- [Serialization](../advanced/serialization.md)
