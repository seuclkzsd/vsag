
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
#include <cstdio>
#include <vector>

#include "cuda_kmeans_assign.h"

namespace vsag::gpu {
namespace {

constexpr uint64_t kMinWorkForGpu = 1ULL << 31;  // 2^31 query*k pairs*dim, see report
constexpr int kThreads = 256;

#define VSAG_CUDA_TRY(expr)                     \
    do {                                        \
        cudaError_t _e = (expr);                \
        if (_e != cudaSuccess) {                \
            return false;                       \
        }                                       \
    } while (0)

__global__ void
init_best(float* best_val, int* best_idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        best_val[i] = 3.402823466e+38F;
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
                      int n,
                      int centroid_offset,
                      int* __restrict__ best_idx,
                      float* __restrict__ best_val) {
    extern __shared__ char raw[];
    float* s_val = reinterpret_cast<float*>(raw);
    int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

    int q = blockIdx.x;
    const float* col = dist + (size_t)q * ld;
    float lv = 3.402823466e+38F;
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

    // Split the budget: centroids resident if they fit, then two query buffers
    // and two distance buffers (double buffering across two streams).
    const uint64_t cent_bytes = k * (uint64_t)dim * sizeof(float);
    const bool cent_resident = cent_bytes * 2 < budget_bytes;
    uint64_t k_chunk = cent_resident ? k : std::min<uint64_t>(k, 8192);
    uint64_t remain = budget_bytes > cent_bytes ? budget_bytes - cent_bytes : budget_bytes / 2;
    // per query row: dim floats (x2 buffers) + k_chunk floats of dist (x2) + 3 scalars
    uint64_t per_row = (uint64_t)dim * 2 * sizeof(float) + k_chunk * 2 * sizeof(float) + 16;
    uint64_t n_chunk = std::max<uint64_t>(1024, std::min<uint64_t>(query_count, remain / per_row));

    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);  // keep FP32: TF32 perturbs labels

    cudaStream_t stream[2] = {nullptr, nullptr};
    float *d_q[2] = {nullptr, nullptr}, *d_dist[2] = {nullptr, nullptr};
    float *d_cent = nullptr, *d_csqr = nullptr, *d_best = nullptr, *d_qsqr = nullptr;
    int* d_bidx = nullptr;
    float *h_q[2] = {nullptr, nullptr};
    int* h_lab = nullptr;
    float* h_val = nullptr;
    bool ok = true;

    auto cleanup = [&]() {
        for (int i = 0; i < 2; ++i) {
            if (d_q[i]) cudaFree(d_q[i]);
            if (d_dist[i]) cudaFree(d_dist[i]);
            if (h_q[i]) cudaFreeHost(h_q[i]);
            if (stream[i]) cudaStreamDestroy(stream[i]);
        }
        if (d_cent) cudaFree(d_cent);
        if (d_csqr) cudaFree(d_csqr);
        if (d_best) cudaFree(d_best);
        if (d_qsqr) cudaFree(d_qsqr);
        if (d_bidx) cudaFree(d_bidx);
        if (h_lab) cudaFreeHost(h_lab);
        if (h_val) cudaFreeHost(h_val);
        if (handle) cublasDestroy(handle);
    };

#define TRY(expr)                    \
    if ((expr) != cudaSuccess) {     \
        ok = false;                  \
        goto done;                   \
    }

    TRY(cudaStreamCreate(&stream[0]));
    TRY(cudaStreamCreate(&stream[1]));
    TRY(cudaMalloc(&d_cent, (cent_resident ? k : k_chunk) * dim * sizeof(float)));
    TRY(cudaMalloc(&d_csqr, (cent_resident ? k : k_chunk) * sizeof(float)));
    TRY(cudaMalloc(&d_best, n_chunk * sizeof(float)));
    TRY(cudaMalloc(&d_qsqr, n_chunk * sizeof(float)));
    TRY(cudaMalloc(&d_bidx, n_chunk * sizeof(int)));
    TRY(cudaHostAlloc(&h_lab, n_chunk * sizeof(int), cudaHostAllocDefault));
    TRY(cudaHostAlloc(&h_val, n_chunk * sizeof(float), cudaHostAllocDefault));
    for (int i = 0; i < 2; ++i) {
        TRY(cudaMalloc(&d_q[i], n_chunk * dim * sizeof(float)));
        TRY(cudaMalloc(&d_dist[i], k_chunk * n_chunk * sizeof(float)));
        TRY(cudaHostAlloc(&h_q[i], n_chunk * dim * sizeof(float), cudaHostAllocDefault));
    }

    if (cent_resident) {
        TRY(cudaMemcpy(d_cent, centroids, cent_bytes, cudaMemcpyHostToDevice));
        row_sqr_norms<<<(int)k, kThreads>>>(d_cent, dim, (int)k, d_csqr);
    }

    {
        double err_acc = 0.0;
        int buf = 0;
        for (uint64_t off = 0; off < query_count; off += n_chunk) {
            uint64_t cur = std::min<uint64_t>(n_chunk, query_count - off);
            cudaStream_t st = stream[buf];
            TRY(cudaStreamSynchronize(st));
            std::copy(query + off * dim, query + (off + cur) * dim, h_q[buf]);
            TRY(cudaMemcpyAsync(d_q[buf], h_q[buf], cur * dim * sizeof(float),
                                cudaMemcpyHostToDevice, st));
            init_best<<<(int)((cur + kThreads - 1) / kThreads), kThreads, 0, st>>>(
                d_best, d_bidx, (int)cur);
            row_sqr_norms<<<(int)cur, kThreads, 0, st>>>(d_q[buf], dim, (int)cur, d_qsqr);

            for (uint64_t koff = 0; koff < k; koff += k_chunk) {
                uint64_t kc = std::min<uint64_t>(k_chunk, k - koff);
                const float* cptr = d_cent;
                const float* sptr = d_csqr;
                if (cent_resident) {
                    cptr = d_cent + koff * dim;
                    sptr = d_csqr + koff;
                } else {
                    TRY(cudaMemcpyAsync(d_cent, centroids + koff * dim,
                                        kc * dim * sizeof(float), cudaMemcpyHostToDevice, st));
                    row_sqr_norms<<<(int)kc, kThreads, 0, st>>>(d_cent, dim, (int)kc, d_csqr);
                }
                const float alpha = -2.0F, beta = 0.0F;
                cublasSetStream(handle, st);
                if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)kc, (int)cur, dim,
                                &alpha, cptr, dim, d_q[buf], dim, &beta, d_dist[buf],
                                (int)kc) != CUBLAS_STATUS_SUCCESS) {
                    ok = false;
                    goto done;
                }
                fused_add_csqr_argmin<<<(int)cur, kThreads,
                                        kThreads * (sizeof(float) + sizeof(int)), st>>>(
                    d_dist[buf], (int)kc, sptr, (int)kc, (int)cur, (int)koff, d_bidx, d_best);
            }
            add_qsqr<<<(int)((cur + kThreads - 1) / kThreads), kThreads, 0, st>>>(
                d_qsqr, d_best, (int)cur);
            TRY(cudaMemcpyAsync(h_lab, d_bidx, cur * sizeof(int), cudaMemcpyDeviceToHost, st));
            TRY(cudaMemcpyAsync(h_val, d_best, cur * sizeof(float), cudaMemcpyDeviceToHost, st));
            TRY(cudaStreamSynchronize(st));
            for (uint64_t i = 0; i < cur; ++i) {
                labels[off + i] = h_lab[i];
                err_acc += h_val[i];
            }
            buf ^= 1;
        }
        TRY(cudaDeviceSynchronize());
        if (error != nullptr) {
            *error = err_acc / (double)query_count;
        }
    }

done:
    cleanup();
#undef TRY
    return ok;
}

}  // namespace vsag::gpu
