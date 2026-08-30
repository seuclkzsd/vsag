
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>

namespace vsag::gpu {

/// True when this build has a CUDA backend and at least one usable device.
bool
CudaAvailable();

/// Chunked GPU nearest-centroid assignment.
///
/// Computes, for every row of `query` (query_count x dim, row-major), the index
/// of the nearest row of `centroids` (k x dim, row-major) under squared L2, and
/// returns the sum of the squared distances through `error`.
///
/// The device working set is bounded by `budget_bytes`; queries and centroids
/// are streamed in chunks so neither operand has to fit in VRAM. Returns false
/// if the CUDA backend is unavailable or the problem is too small to be worth
/// offloading, in which case the caller must use the CPU path.
bool
CudaAssignNearest(const float* query,
                  uint64_t query_count,
                  const float* centroids,
                  uint64_t k,
                  int32_t dim,
                  int32_t* labels,
                  double* error,
                  uint64_t budget_bytes);

/// GPU k-means++ seeding.
///
/// Chooses `k` initial centroids from `datas` (count x dim, row-major) with the
/// standard D^2 rule and writes them to `centroids_out` (k x dim, row-major).
/// `uniforms` supplies 2*k host-drawn values in [0, 1): the pair at 2*c is the
/// weighted pick for centroid c and its fallback for a degenerate (all-zero)
/// weight vector.
///
/// The whole dataset has to stay resident on the device, because every one of
/// the k steps sweeps all of it; streaming it back per step would cost more
/// than the CPU path. Returns false when it does not fit in `budget_bytes`, or
/// when the backend is unavailable, in which case the caller must seed on the
/// CPU.
///
/// The selection is statistically equivalent to the CPU routine but not
/// identical: the prefix sums are accumulated in a different order, so a given
/// draw can land on a neighbouring point.
bool
CudaKMeansPlusPlusInit(const float* datas,
                       uint64_t count,
                       int32_t dim,
                       uint32_t k,
                       const float* uniforms,
                       float* centroids_out,
                       uint64_t budget_bytes);

/// Sums every point into the accumulator of the centroid it was assigned to.
///
/// Writes the per-centroid sums to `sums` (k x dim, row-major) and the number
/// of points that landed on each centroid to `counts` (k), both overwritten
/// rather than accumulated. The caller divides one by the other and decides
/// what to do with empty clusters.
///
/// Points are streamed in chunks, so nothing has to fit in VRAM beyond the
/// accumulators themselves. Returns false if the backend is unavailable or the
/// accumulators do not fit in `budget_bytes`, in which case the caller must
/// accumulate on the CPU.
///
/// The sums are built with floating-point atomics, so their rounding depends on
/// the order the additions happen to land in and differs slightly from a serial
/// accumulation.
bool
CudaAccumulateCentroids(const float* datas,
                        uint64_t count,
                        int32_t dim,
                        const int32_t* labels,
                        uint32_t k,
                        float* sums,
                        int32_t* counts,
                        uint64_t budget_bytes);

/// Device working-set budget derived from what the selected device actually has
/// free, rather than from a fixed guess.
///
/// Leaves a fifth of the free memory to allocator fragmentation, the CUDA
/// context's own growth, and anything else sharing the device, so a build stays
/// well inside the ceiling the callers target. Returns 0 when no usable device
/// is present, which every entry point below treats as "do not offload".
uint64_t
CudaSuggestedBudget();

}  // namespace vsag::gpu
