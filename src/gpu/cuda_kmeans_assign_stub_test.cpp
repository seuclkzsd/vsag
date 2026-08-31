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

#include <vector>

#include "cuda_kmeans_assign.h"
#include "unittest.h"

// The contract every caller relies on: with the backend compiled out, each
// entry point refuses and the caller keeps its CPU path. Nothing here needs a
// device, which is the point -- CI has none.
#ifndef VSAG_ENABLE_CUDA

TEST_CASE("Compiled out, the backend reports no device", "[ut][gpu_stub]") {
    REQUIRE_FALSE(vsag::gpu::CudaAvailable());
    REQUIRE_FALSE(vsag::gpu::CudaSelectDevice(0));
    REQUIRE_FALSE(vsag::gpu::CudaSelectDevice(99));
    // A zero budget is what every caller then passes on, so the entry points
    // have to treat it as "do not offload" rather than sizing a plan around it.
    REQUIRE(vsag::gpu::CudaSuggestedBudget(0) == 0);
    REQUIRE(vsag::gpu::CudaSuggestedBudget(2ULL << 30) == 0);
}

TEST_CASE("Compiled out, every offload refuses", "[ut][gpu_stub]") {
    const uint64_t count = 4096;
    const int32_t dim = 8;
    const uint32_t k = 16;
    std::vector<float> data(count * dim, 0.5F);
    std::vector<float> centroids((uint64_t)k * dim, 0.25F);
    std::vector<int32_t> labels(count, -1);
    std::vector<float> uniforms(2ULL * k, 0.5F);
    std::vector<float> sums((uint64_t)k * dim, 0.0F);
    std::vector<int32_t> counts(k, 0);
    double error = 0.0;

    REQUIRE_FALSE(vsag::gpu::CudaAssignNearest(data.data(),
                                               count,
                                               centroids.data(),
                                               k,
                                               dim,
                                               labels.data(),
                                               &error,
                                               2ULL << 30,
                                               0));
    REQUIRE_FALSE(vsag::gpu::CudaKMeansPlusPlusInit(
        data.data(), count, dim, k, uniforms.data(), centroids.data(), 2ULL << 30));
    REQUIRE_FALSE(vsag::gpu::CudaAccumulateCentroids(
        data.data(), count, dim, labels.data(), k, sums.data(), counts.data(), 2ULL << 30));

    // A refusal must leave the caller's buffers alone, since the caller is about
    // to fill them from the CPU path.
    REQUIRE(labels.front() == -1);
    REQUIRE(counts.front() == 0);
}

#endif  // VSAG_ENABLE_CUDA
