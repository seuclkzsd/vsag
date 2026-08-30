
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

}  // namespace vsag::gpu
