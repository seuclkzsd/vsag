
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

// Built when ENABLE_CUDA is OFF so callers can link unconditionally and take
// the CPU path at run time.

#include "cuda_kmeans_assign.h"

namespace vsag::gpu {

bool
CudaAvailable() {
    return false;
}

bool
CudaAssignNearest(const float* /*query*/,
                  uint64_t /*query_count*/,
                  const float* /*centroids*/,
                  uint64_t /*k*/,
                  int32_t /*dim*/,
                  int32_t* /*labels*/,
                  double* /*error*/,
                  uint64_t /*budget_bytes*/,
                  uint64_t /*min_work*/) {
    return false;
}

bool
CudaKMeansPlusPlusInit(const float* /*datas*/,
                       uint64_t /*count*/,
                       int32_t /*dim*/,
                       uint32_t /*k*/,
                       const float* /*uniforms*/,
                       float* /*centroids_out*/,
                       uint64_t /*budget_bytes*/) {
    return false;
}

bool
CudaAccumulateCentroids(const float* /*datas*/,
                        uint64_t /*count*/,
                        int32_t /*dim*/,
                        const int32_t* /*labels*/,
                        uint32_t /*k*/,
                        float* /*sums*/,
                        int32_t* /*counts*/,
                        uint64_t /*budget_bytes*/) {
    return false;
}

uint64_t
CudaSuggestedBudget(uint64_t /*cap_bytes*/) {
    return 0;
}

bool
CudaSelectDevice(int32_t /*device_id*/) {
    return false;
}

}  // namespace vsag::gpu
