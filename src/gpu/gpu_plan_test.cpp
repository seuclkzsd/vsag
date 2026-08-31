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

#include <algorithm>

#include "gpu_plan.h"

#include "unittest.h"

using namespace vsag::gpu;

namespace {
constexpr uint64_t kGiB = 1ULL << 30;
constexpr uint64_t kBigWork = 1ULL << 20;  // rows and centroids well past the floor
}  // namespace

TEST_CASE("PlanAssign refuses degenerate input", "[ut][gpu_plan]") {
    REQUIRE_FALSE(PlanAssign(0, 4096, 128, 2 * kGiB, 0).offload);
    REQUIRE_FALSE(PlanAssign(kBigWork, 0, 128, 2 * kGiB, 0).offload);
    REQUIRE_FALSE(PlanAssign(kBigWork, 4096, 0, 2 * kGiB, 0).offload);
    REQUIRE_FALSE(PlanAssign(kBigWork, 4096, -1, 2 * kGiB, 0).offload);
    // A zero budget is what CudaSuggestedBudget reports with no usable device.
    REQUIRE_FALSE(PlanAssign(kBigWork, 4096, 128, 0, 0).offload);
}

TEST_CASE("PlanAssign keeps small problems on the CPU", "[ut][gpu_plan]") {
    // The floor is on query_count * k * dim, so a small problem is refused
    // however the smallness is distributed.
    REQUIRE_FALSE(PlanAssign(1024, 16, 128, 2 * kGiB, 0).offload);
    REQUIRE(PlanAssign(262144, 4096, 128, 2 * kGiB, 0).offload);

    // Raising the threshold pushes a problem back to the CPU, lowering it
    // pulls one onto the device. Both directions matter: the first is how a
    // user opts out, the second is how they opt in on faster hardware.
    const uint64_t work = 262144ULL * 4096ULL * 128ULL;
    REQUIRE_FALSE(PlanAssign(262144, 4096, 128, 2 * kGiB, work * 2).offload);
    REQUIRE(PlanAssign(1024, 16, 128, 2 * kGiB, 1).offload);
}

TEST_CASE("PlanAssign keeps centroids resident when they fit", "[ut][gpu_plan]") {
    // 4096 centroids of 128 dimensions is 2 MiB, trivially resident in 2 GiB.
    const auto small = PlanAssign(262144, 4096, 128, 2 * kGiB, 0);
    REQUIRE(small.offload);
    REQUIRE(small.centroids_resident);
    REQUIRE(small.k_chunk == 4096);
    REQUIRE(small.centroid_slots == 1);

    // A centroid set larger than half the budget has to be streamed, and then
    // each stream needs its own slice so the two cannot clobber each other.
    const auto streamed = PlanAssign(kBigWork, 1u << 21, 128, 512ULL << 20, 0);
    REQUIRE(streamed.offload);
    REQUIRE_FALSE(streamed.centroids_resident);
    REQUIRE(streamed.k_chunk == kStreamedCentroidChunk);
    REQUIRE(streamed.centroid_slots == 2);
}

TEST_CASE("PlanAssign chunks stay within the budget", "[ut][gpu_plan]") {
    // The working set the plan implies must not exceed what it was given. This
    // is the property that keeps peak device memory under control.
    for (const uint64_t budget : {kGiB / 4, kGiB, 2 * kGiB, 8 * kGiB}) {
        for (const uint64_t k : {1024ULL, 4096ULL, 40000ULL}) {
            const auto plan = PlanAssign(2560000, k, 128, budget, 0);
            if (not plan.offload) {
                continue;
            }
            const uint64_t per_row = 128ULL * 2 * sizeof(float) +
                                     plan.k_chunk * 2 * sizeof(float) + 24;
            // A plan that offloads must fit the budget it was given. No
            // exemption: a floor that pushes the working set past the budget is
            // a reason to refuse, not to overspend.
            REQUIRE(plan.n_chunk * per_row <= budget);
            REQUIRE(plan.n_chunk >= std::min<uint64_t>(2560000, kMinChunkRows));
            REQUIRE(plan.n_chunk <= 2560000);
        }
    }
}

TEST_CASE("PlanAssign refuses a budget it cannot honour", "[ut][gpu_plan]") {
    // 4096 centroids of 128 dimensions need 2 MiB just for the centroids, and a
    // 1024-row chunk another 33 MiB. Under a 1 MiB budget the pass must refuse:
    // launching anyway would use 33x what the caller allowed.
    REQUIRE_FALSE(PlanAssign(262144, 4096, 128, 1ULL << 20, 0).offload);
    REQUIRE_FALSE(PlanAccumulate(262144, 128, 4096, 1ULL << 20).offload);
    // Enough room for the floor-sized chunk, so it goes ahead.
    REQUIRE(PlanAssign(262144, 4096, 128, 64ULL << 20, 0).offload);
}

TEST_CASE("PlanAssign never asks for more rows than it has", "[ut][gpu_plan]") {
    const auto plan = PlanAssign(262144, 4096, 128, 64 * kGiB, 0);
    REQUIRE(plan.offload);
    REQUIRE(plan.n_chunk == 262144);
}

TEST_CASE("PlanSeed refuses what cannot stay resident", "[ut][gpu_plan]") {
    // Seeding sweeps the whole dataset once per centroid, so the dataset has to
    // fit; when it does not the caller must seed on the CPU.
    REQUIRE(PlanSeed(262144, 128, 4096, 2 * kGiB).offload);
    REQUIRE_FALSE(PlanSeed(262144, 128, 4096, 64ULL << 20).offload);

    // 1048576 points of 960 dimensions is 3.75 GiB: refused under a 2 GiB
    // budget, accepted once the budget reflects a 24 GiB card. This is the case
    // that used to fall back and cost hours.
    REQUIRE_FALSE(PlanSeed(1048576, 960, 16384, 2 * kGiB).offload);
    REQUIRE(PlanSeed(1048576, 960, 16384, 19 * kGiB).offload);
}

TEST_CASE("PlanSeed refuses degenerate input", "[ut][gpu_plan]") {
    REQUIRE_FALSE(PlanSeed(0, 128, 16, 2 * kGiB).offload);
    REQUIRE_FALSE(PlanSeed(1024, 128, 0, 2 * kGiB).offload);
    REQUIRE_FALSE(PlanSeed(1024, 0, 16, 2 * kGiB).offload);
    // k centroids cannot be drawn from fewer than k points.
    REQUIRE_FALSE(PlanSeed(16, 128, 32, 2 * kGiB).offload);
}

TEST_CASE("PlanSeed tiles cover the dataset and fit shared memory", "[ut][gpu_plan]") {
    for (const uint64_t count : {1024ULL, 65536ULL, 262144ULL, 1048576ULL}) {
        const auto plan = PlanSeed(count, 128, 512, 19 * kGiB);
        REQUIRE(plan.offload);
        REQUIRE(plan.blocks > 0);
        // Every point must land in exactly one tile.
        REQUIRE((uint64_t)plan.blocks * plan.rows_per_block >= count);
        REQUIRE(((uint64_t)plan.blocks - 1) * plan.rows_per_block < count);
        REQUIRE(plan.shared_bytes <= kMaxInitShared);
    }
    // Fewer points than tiles gives one point per tile rather than empty ones.
    const auto tiny = PlanSeed(64, 128, 8, 2 * kGiB);
    REQUIRE(tiny.offload);
    REQUIRE(tiny.blocks == 64);
    REQUIRE(tiny.rows_per_block == 1);
}

TEST_CASE("PlanAccumulate refuses when the accumulators do not fit", "[ut][gpu_plan]") {
    REQUIRE(PlanAccumulate(262144, 128, 4096, 2 * kGiB).offload);
    // 40000 centroids of 128 dimensions is 20.5 MiB of sums; a 16 MiB budget
    // cannot hold them whatever the chunk size.
    REQUIRE_FALSE(PlanAccumulate(262144, 128, 40000, 16ULL << 20).offload);
    REQUIRE_FALSE(PlanAccumulate(0, 128, 4096, 2 * kGiB).offload);
    REQUIRE_FALSE(PlanAccumulate(262144, 128, 0, 2 * kGiB).offload);
}

TEST_CASE("PlanAccumulate chunks stay within the budget", "[ut][gpu_plan]") {
    for (const uint64_t budget : {kGiB / 8, kGiB, 2 * kGiB}) {
        const auto plan = PlanAccumulate(2560000, 128, 4096, budget);
        REQUIRE(plan.offload);
        const uint64_t per_row = 128ULL * sizeof(float) + sizeof(int32_t);
        REQUIRE(plan.accumulator_bytes + plan.n_chunk * per_row <= budget);
        REQUIRE(plan.n_chunk <= 2560000);
    }
}

TEST_CASE("CapBudget leaves headroom and honours the ceiling", "[ut][gpu_plan]") {
    // A fifth of what is free stays behind for fragmentation and the context.
    REQUIRE(CapBudget(10 * kGiB, 0) == (uint64_t)(10.0 * (double)kGiB * 0.8));
    // A ceiling of 0 means no ceiling; otherwise the smaller of the two wins.
    REQUIRE(CapBudget(24 * kGiB, kChunkedBudgetCap) == kChunkedBudgetCap);
    REQUIRE(CapBudget(kGiB, kChunkedBudgetCap) == (uint64_t)((double)kGiB * 0.8));
    REQUIRE(CapBudget(0, kChunkedBudgetCap) == 0);
}

TEST_CASE("A zero budget keeps every path on the CPU", "[ut][gpu_plan]") {
    // CudaSuggestedBudget returns 0 when there is no usable device, and each
    // entry point has to treat that as "do not offload" rather than trying to
    // size a plan around it.
    REQUIRE_FALSE(PlanAssign(kBigWork, 4096, 128, 0, 0).offload);
    REQUIRE_FALSE(PlanSeed(262144, 128, 4096, 0).offload);
    REQUIRE_FALSE(PlanAccumulate(262144, 128, 4096, 0).offload);
}
