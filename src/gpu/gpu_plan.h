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

#include <algorithm>
#include <cstdint>

// Sizing decisions for the CUDA backend, kept free of CUDA so they can be
// tested on a machine without a device. Each entry point in
// cuda_kmeans_assign.h asks for a plan first and refuses the work when the plan
// says the problem does not fit or is not worth offloading.

namespace vsag::gpu {

/// Below this many query * centroid * dimension products the launch and
/// transfer overhead outweighs the device. Measured against the CPU path on a
/// single RTX 3090; the crossover sat near k = 2000-4000 for 128 dimensions.
constexpr uint64_t kMinWorkForGpu = 1ULL << 31;

/// Centroid slice size when the full set cannot stay resident.
constexpr uint64_t kStreamedCentroidChunk = 8192;

/// Tiles the seeding pass splits the dataset into, so a weighted draw only has
/// to walk the tile sums plus the one tile it lands in.
constexpr int32_t kInitBlocks = 1024;

/// Ceiling on the shared memory kpp_select stages, kept clear of the 48 KiB
/// limit.
constexpr uint64_t kMaxInitShared = 44U << 10;

/// Working-set ceiling for the chunked paths. A sweep over chunk sizes put the
/// best runtime near a 1.1 GiB working set and found 10.5 GiB slower, so a
/// larger share of the card buys nothing and makes every allocation dearer.
constexpr uint64_t kChunkedBudgetCap = 2ULL << 30;

/// Smallest chunk worth issuing, so a tight budget still produces a launch with
/// enough rows to fill the device.
constexpr uint64_t kMinChunkRows = 1024;

/// How one nearest-centroid pass gets cut up.
struct AssignPlan {
    bool offload{false};  ///< false: the caller must use the CPU path
    bool centroids_resident{false};
    uint64_t centroid_slots{0};  ///< streamed centroids need one slice per stream
    uint64_t k_chunk{0};
    uint64_t n_chunk{0};
};

/// `min_work` of 0 selects kMinWorkForGpu.
inline AssignPlan
PlanAssign(uint64_t query_count, uint64_t k, int32_t dim, uint64_t budget_bytes,
           uint64_t min_work) {
    AssignPlan plan;
    if (query_count == 0 || k == 0 || dim <= 0 || budget_bytes == 0) {
        return plan;
    }
    const uint64_t work_floor = min_work > 0 ? min_work : kMinWorkForGpu;
    if (query_count * k < work_floor / (uint64_t)dim) {
        return plan;
    }

    const uint64_t cent_bytes = k * (uint64_t)dim * sizeof(float);
    plan.centroids_resident = cent_bytes * 2 < budget_bytes;
    plan.k_chunk = plan.centroids_resident ? k : std::min<uint64_t>(k, kStreamedCentroidChunk);
    plan.centroid_slots = plan.centroids_resident ? 1 : 2;

    const uint64_t remain =
        budget_bytes > cent_bytes ? budget_bytes - cent_bytes : budget_bytes / 2;
    // Per query row across both slots: the row itself, its distances, and the
    // best value, its index and the row's squared norm.
    const uint64_t per_row =
        (uint64_t)dim * 2 * sizeof(float) + plan.k_chunk * 2 * sizeof(float) + 24;
    plan.n_chunk =
        std::max<uint64_t>(kMinChunkRows, std::min<uint64_t>(query_count, remain / per_row));
    plan.offload = true;
    return plan;
}

/// How the k-means++ seeding pass gets laid out.
struct SeedPlan {
    bool offload{false};
    int32_t blocks{0};
    uint64_t rows_per_block{0};
    uint64_t shared_bytes{0};
    uint64_t device_bytes{0};  ///< dataset, running minima, tile sums and picks
};

inline SeedPlan
PlanSeed(uint64_t count, int32_t dim, uint32_t k, uint64_t budget_bytes) {
    SeedPlan plan;
    if (count == 0 || k == 0 || dim <= 0 || (uint64_t)k > count) {
        return plan;
    }
    plan.blocks = (int32_t)std::min<uint64_t>((uint64_t)kInitBlocks, count);
    plan.rows_per_block = (count + (uint64_t)plan.blocks - 1) / (uint64_t)plan.blocks;
    plan.shared_bytes = ((uint64_t)plan.blocks + plan.rows_per_block) * sizeof(float);
    if (plan.shared_bytes > kMaxInitShared) {
        return plan;
    }
    // The dataset stays resident: every one of the k steps reads all of it, so
    // streaming it back per step would cost more than the CPU path.
    plan.device_bytes = count * (uint64_t)dim * sizeof(float) + count * sizeof(float) +
                        (uint64_t)plan.blocks * sizeof(float) + (uint64_t)k * sizeof(uint64_t);
    if (plan.device_bytes > budget_bytes) {
        return plan;
    }
    plan.offload = true;
    return plan;
}

/// How the centroid accumulation pass gets cut up.
struct AccumulatePlan {
    bool offload{false};
    uint64_t n_chunk{0};
    uint64_t accumulator_bytes{0};
};

inline AccumulatePlan
PlanAccumulate(uint64_t count, int32_t dim, uint32_t k, uint64_t budget_bytes) {
    AccumulatePlan plan;
    if (count == 0 || k == 0 || dim <= 0) {
        return plan;
    }
    const uint64_t sum_bytes = (uint64_t)k * (uint64_t)dim * sizeof(float);
    const uint64_t cnt_bytes = (uint64_t)k * sizeof(int32_t);
    plan.accumulator_bytes = sum_bytes + cnt_bytes;
    if (plan.accumulator_bytes >= budget_bytes) {
        return plan;
    }
    const uint64_t per_row = (uint64_t)dim * sizeof(float) + sizeof(int32_t);
    const uint64_t remain = budget_bytes - plan.accumulator_bytes;
    plan.n_chunk =
        std::max<uint64_t>(kMinChunkRows, std::min<uint64_t>(count, remain / per_row));
    plan.offload = true;
    return plan;
}

/// Share of free device memory a build may take, leaving the rest to allocator
/// fragmentation, the CUDA context's own growth, and anything else on the card.
/// `cap_bytes` of 0 means no ceiling.
inline uint64_t
CapBudget(uint64_t free_bytes, uint64_t cap_bytes) {
    const uint64_t usable = (uint64_t)((double)free_bytes * 0.8);
    return cap_bytes > 0 ? std::min<uint64_t>(usable, cap_bytes) : usable;
}

}  // namespace vsag::gpu
