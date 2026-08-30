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

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "cuda_kmeans_assign.h"

namespace vsag::gpu {
namespace {

constexpr uint64_t kMinWorkForGpu = 1ULL << 31;  // n * k * dim below this: CPU wins
constexpr int kThreads = 256;
constexpr float kFloatMax = 3.402823466e+38F;

__global__ void
init_best(float* best_val, int* best_idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        best_val[i] = kFloatMax;
        best_idx[i] = -1;
    }
}

__global__ void
row_sqr_norms(const float* __restrict__ x, int dim, int rows, float* __restrict__ out) {
    int r = blockIdx.x;
    const float* p = x + (size_t)r * dim;
    float acc = 0.F;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        acc += p[i] * p[i];
    }
    __shared__ float s[kThreads];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int t = blockDim.x / 2; t > 0; t >>= 1) {
        if (threadIdx.x < t) {
            s[threadIdx.x] += s[threadIdx.x + t];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[r] = s[0];
    }
}

// dist is column-major (kc rows x n cols). Fuses "+ ||c||^2" into the argmin so
// the k x n matrix is read exactly once and never leaves the device.
__global__ void
fused_add_csqr_argmin(const float* __restrict__ dist,
                      int ld,
                      const float* __restrict__ c_sqr,
                      int kc,
                      int centroid_offset,
                      int* __restrict__ best_idx,
                      float* __restrict__ best_val) {
    extern __shared__ char raw[];
    float* s_val = reinterpret_cast<float*>(raw);
    int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

    int q = blockIdx.x;
    const float* col = dist + (size_t)q * ld;
    float lv = kFloatMax;
    int li = -1;
    for (int c = threadIdx.x; c < kc; c += blockDim.x) {
        float v = col[c] + c_sqr[c];
        if (v < lv) {
            lv = v;
            li = c + centroid_offset;
        }
    }
    s_val[threadIdx.x] = lv;
    s_idx[threadIdx.x] = li;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && s_val[threadIdx.x + s] < s_val[threadIdx.x]) {
            s_val[threadIdx.x] = s_val[threadIdx.x + s];
            s_idx[threadIdx.x] = s_idx[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && s_val[0] < best_val[q]) {
        best_val[q] = s_val[0];
        best_idx[q] = s_idx[0];
    }
}

__global__ void
add_qsqr(const float* __restrict__ q_sqr, float* __restrict__ best_val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        best_val[i] += q_sqr[i];
    }
}

}  // namespace

bool
CudaAvailable() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return false;
    }
    return count > 0;
}

bool
CudaAssignNearest(const float* query,
                  uint64_t query_count,
                  const float* centroids,
                  uint64_t k,
                  int32_t dim,
                  int32_t* labels,
                  double* error,
                  uint64_t budget_bytes) {
    if (query == nullptr || centroids == nullptr || labels == nullptr || query_count == 0 ||
        k == 0 || dim <= 0) {
        return false;
    }
    if (query_count * k < kMinWorkForGpu / (uint64_t)dim) {
        return false;  // too small: CPU wins, see the crossover measurement
    }
    if (!CudaAvailable()) {
        return false;
    }

    // Centroids stay resident when they fit; otherwise they are streamed in
    // k_chunk slices. Both slots of every per-stream buffer are sized here so
    // the two streams never touch the same memory -- that is what lets the
    // transfer of one chunk overlap the compute of the previous one.
    const uint64_t cent_bytes = k * (uint64_t)dim * sizeof(float);
    const bool cent_resident = cent_bytes * 2 < budget_bytes;
    const uint64_t k_chunk = cent_resident ? k : std::min<uint64_t>(k, 8192);
    const uint64_t cent_slots = cent_resident ? 1 : 2;  // streamed centroids need one per stream
    const uint64_t cent_alloc = cent_resident ? k : k_chunk;
    uint64_t remain = budget_bytes > cent_bytes ? budget_bytes - cent_bytes : budget_bytes / 2;
    // Per query row, both slots together: query 2*dim floats, distances
    // 2*k_chunk floats, and 2*(best + qsqr + bidx) = 24 bytes of scalars.
    const uint64_t per_row =
        (uint64_t)dim * 2 * sizeof(float) + k_chunk * 2 * sizeof(float) + 24;
    const uint64_t n_chunk =
        std::max<uint64_t>(1024, std::min<uint64_t>(query_count, remain / per_row));

    cublasHandle_t handle[2] = {nullptr, nullptr};
    cudaStream_t stream[2] = {nullptr, nullptr};
    float* d_q[2] = {nullptr, nullptr};
    float* d_dist[2] = {nullptr, nullptr};
    float* d_best[2] = {nullptr, nullptr};
    float* d_qsqr[2] = {nullptr, nullptr};
    int* d_bidx[2] = {nullptr, nullptr};
    float* d_cent[2] = {nullptr, nullptr};
    float* d_csqr[2] = {nullptr, nullptr};
    float* h_q[2] = {nullptr, nullptr};
    int* h_lab[2] = {nullptr, nullptr};
    float* h_val[2] = {nullptr, nullptr};

    // Chunk currently in flight on each stream, waiting to be copied out.
    uint64_t pend_off[2] = {0, 0};
    uint64_t pend_cnt[2] = {0, 0};
    double err_acc = 0.0;
    bool ok = true;

    auto cleanup = [&]() {
        for (int i = 0; i < 2; ++i) {
            if (d_q[i] != nullptr) cudaFree(d_q[i]);
            if (d_dist[i] != nullptr) cudaFree(d_dist[i]);
            if (d_best[i] != nullptr) cudaFree(d_best[i]);
            if (d_qsqr[i] != nullptr) cudaFree(d_qsqr[i]);
            if (d_bidx[i] != nullptr) cudaFree(d_bidx[i]);
            if (d_cent[i] != nullptr) cudaFree(d_cent[i]);
            if (d_csqr[i] != nullptr) cudaFree(d_csqr[i]);
            if (h_q[i] != nullptr) cudaFreeHost(h_q[i]);
            if (h_lab[i] != nullptr) cudaFreeHost(h_lab[i]);
            if (h_val[i] != nullptr) cudaFreeHost(h_val[i]);
            if (handle[i] != nullptr) cublasDestroy(handle[i]);
            if (stream[i] != nullptr) cudaStreamDestroy(stream[i]);
        }
    };

    // Copies one finished chunk's labels and distances out of pinned memory.
    // The caller must have synchronized that slot's stream first.
    auto drain = [&](int slot) {
        for (uint64_t i = 0; i < pend_cnt[slot]; ++i) {
            labels[pend_off[slot] + i] = h_lab[slot][i];
            err_acc += h_val[slot][i];
        }
        pend_cnt[slot] = 0;
    };

#define TRY(expr)                \
    if ((expr) != cudaSuccess) { \
        ok = false;              \
        goto done;               \
    }

    for (int i = 0; i < 2; ++i) {
        TRY(cudaStreamCreate(&stream[i]));
        TRY(cudaMalloc(&d_q[i], n_chunk * dim * sizeof(float)));
        TRY(cudaMalloc(&d_dist[i], k_chunk * n_chunk * sizeof(float)));
        TRY(cudaMalloc(&d_best[i], n_chunk * sizeof(float)));
        TRY(cudaMalloc(&d_qsqr[i], n_chunk * sizeof(float)));
        TRY(cudaMalloc(&d_bidx[i], n_chunk * sizeof(int)));
        TRY(cudaHostAlloc(&h_q[i], n_chunk * dim * sizeof(float), cudaHostAllocDefault));
        TRY(cudaHostAlloc(&h_lab[i], n_chunk * sizeof(int), cudaHostAllocDefault));
        TRY(cudaHostAlloc(&h_val[i], n_chunk * sizeof(float), cudaHostAllocDefault));
        // One cuBLAS handle per stream: a shared handle serialises its internal
        // workspace across the two streams and would defeat the overlap.
        if (cublasCreate(&handle[i]) != CUBLAS_STATUS_SUCCESS) {
            ok = false;
            goto done;
        }
        cublasSetMathMode(handle[i], CUBLAS_PEDANTIC_MATH);  // keep FP32: TF32 perturbs labels
        cublasSetStream(handle[i], stream[i]);
    }
    for (uint64_t s = 0; s < cent_slots; ++s) {
        TRY(cudaMalloc(&d_cent[s], cent_alloc * dim * sizeof(float)));
        TRY(cudaMalloc(&d_csqr[s], cent_alloc * sizeof(float)));
    }

    if (cent_resident) {
        TRY(cudaMemcpyAsync(
            d_cent[0], centroids, cent_bytes, cudaMemcpyHostToDevice, stream[0]));
        row_sqr_norms<<<(int)k, kThreads, 0, stream[0]>>>(d_cent[0], dim, (int)k, d_csqr[0]);
        // Both streams read these, so the upload has to land before either starts.
        TRY(cudaStreamSynchronize(stream[0]));
    }

    for (uint64_t off = 0, chunk = 0; off < query_count; off += n_chunk, ++chunk) {
        const uint64_t cur = std::min<uint64_t>(n_chunk, query_count - off);
        const int buf = (int)(chunk & 1U);
        const cudaStream_t st = stream[buf];

        // Wait only for the chunk that used this slot two iterations ago; the
        // chunk on the other stream keeps running.
        TRY(cudaStreamSynchronize(st));
        drain(buf);

        std::copy(query + off * dim, query + (off + cur) * dim, h_q[buf]);
        TRY(cudaMemcpyAsync(
            d_q[buf], h_q[buf], cur * dim * sizeof(float), cudaMemcpyHostToDevice, st));
        init_best<<<(int)((cur + kThreads - 1) / kThreads), kThreads, 0, st>>>(
            d_best[buf], d_bidx[buf], (int)cur);
        row_sqr_norms<<<(int)cur, kThreads, 0, st>>>(d_q[buf], dim, (int)cur, d_qsqr[buf]);

        for (uint64_t koff = 0; koff < k; koff += k_chunk) {
            const uint64_t kc = std::min<uint64_t>(k_chunk, k - koff);
            const float* cptr = nullptr;
            const float* sptr = nullptr;
            if (cent_resident) {
                cptr = d_cent[0] + koff * dim;
                sptr = d_csqr[0] + koff;
            } else {
                // Each stream owns its own centroid slice, so the two uploads
                // cannot clobber each other.
                TRY(cudaMemcpyAsync(d_cent[buf],
                                    centroids + koff * dim,
                                    kc * dim * sizeof(float),
                                    cudaMemcpyHostToDevice,
                                    st));
                row_sqr_norms<<<(int)kc, kThreads, 0, st>>>(
                    d_cent[buf], dim, (int)kc, d_csqr[buf]);
                cptr = d_cent[buf];
                sptr = d_csqr[buf];
            }
            const float alpha = -2.0F;
            const float beta = 0.0F;
            if (cublasSgemm(handle[buf],
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            (int)kc,
                            (int)cur,
                            dim,
                            &alpha,
                            cptr,
                            dim,
                            d_q[buf],
                            dim,
                            &beta,
                            d_dist[buf],
                            (int)kc) != CUBLAS_STATUS_SUCCESS) {
                ok = false;
                goto done;
            }
            fused_add_csqr_argmin<<<(int)cur,
                                    kThreads,
                                    kThreads * (sizeof(float) + sizeof(int)),
                                    st>>>(
                d_dist[buf], (int)kc, sptr, (int)kc, (int)koff, d_bidx[buf], d_best[buf]);
        }
        add_qsqr<<<(int)((cur + kThreads - 1) / kThreads), kThreads, 0, st>>>(
            d_qsqr[buf], d_best[buf], (int)cur);
        TRY(cudaMemcpyAsync(
            h_lab[buf], d_bidx[buf], cur * sizeof(int), cudaMemcpyDeviceToHost, st));
        TRY(cudaMemcpyAsync(
            h_val[buf], d_best[buf], cur * sizeof(float), cudaMemcpyDeviceToHost, st));
        pend_off[buf] = off;
        pend_cnt[buf] = cur;
        // No synchronize here: the next chunk is issued on the other stream and
        // its host-to-device copy overlaps this chunk's compute.
    }

    for (int buf = 0; buf < 2; ++buf) {
        TRY(cudaStreamSynchronize(stream[buf]));
        drain(buf);
    }
    if (error != nullptr) {
        *error = err_acc / (double)query_count;
    }

done:
    cleanup();
#undef TRY
    return ok;
}

}  // namespace vsag::gpu
