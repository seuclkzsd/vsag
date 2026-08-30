
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

#include "ivf_partition_strategy_parameter.h"

#include <fmt/format.h>

#include <iostream>

#include "inner_string_params.h"
#include "utils/param_compat_macros.h"
#include "vsag/constants.h"

namespace vsag {

IVFPartitionStrategyParameters::IVFPartitionStrategyParameters() = default;

void
IVFPartitionStrategyParameters::FromJson(const JsonType& json) {
    if (json[IVF_TRAIN_TYPE_KEY].GetString() == IVF_TRAIN_TYPE_KMEANS) {
        this->partition_train_type = IVFNearestPartitionTrainerType::KMeansTrainer;
    } else if (json[IVF_TRAIN_TYPE_KEY].GetString() == IVF_TRAIN_TYPE_RANDOM) {
        this->partition_train_type = IVFNearestPartitionTrainerType::RandomTrainer;
    }

    if (json[IVF_PARTITION_STRATEGY_TYPE_KEY].GetString() == IVF_PARTITION_STRATEGY_TYPE_NEAREST) {
        this->partition_strategy_type = IVFPartitionStrategyType::IVF;
    } else if (json[IVF_PARTITION_STRATEGY_TYPE_KEY].GetString() ==
               IVF_PARTITION_STRATEGY_TYPE_GNO_IMI) {
        this->partition_strategy_type = IVFPartitionStrategyType::GNO_IMI;
    }
    if (json.Contains(IVF_ROUTE_MAX_DEGREE_KEY)) {
        this->route_max_degree = static_cast<int32_t>(json[IVF_ROUTE_MAX_DEGREE_KEY].GetInt());
        CHECK_ARGUMENT(this->route_max_degree > 0, "route_max_degree must be positive");
    }
    if (json.Contains(IVF_ROUTE_EF_CONSTRUCTION_KEY)) {
        this->route_ef_construction =
            static_cast<int32_t>(json[IVF_ROUTE_EF_CONSTRUCTION_KEY].GetInt());
        CHECK_ARGUMENT(this->route_ef_construction > 0, "route_ef_construction must be positive");
    }
    if (json.Contains(IVF_ENABLE_GPU_BUILD_KEY)) {
        this->enable_gpu_build = json[IVF_ENABLE_GPU_BUILD_KEY].GetBool();
    }
    if (json.Contains(IVF_GPU_DEVICE_ID_KEY)) {
        this->gpu_device_id = static_cast<int32_t>(json[IVF_GPU_DEVICE_ID_KEY].GetInt());
        CHECK_ARGUMENT(this->gpu_device_id >= 0, "gpu_device_id must not be negative");
    }
    if (json.Contains(IVF_GPU_MEMORY_BUDGET_KEY)) {
        const auto budget = json[IVF_GPU_MEMORY_BUDGET_KEY].GetInt();
        CHECK_ARGUMENT(budget >= 0, "gpu_memory_budget must not be negative");
        this->gpu_memory_budget = static_cast<uint64_t>(budget);
    }
    if (json.Contains(IVF_GPU_MIN_WORK_THRESHOLD_KEY)) {
        const auto threshold = json[IVF_GPU_MIN_WORK_THRESHOLD_KEY].GetInt();
        CHECK_ARGUMENT(threshold >= 0, "gpu_min_work_threshold must not be negative");
        this->gpu_min_work_threshold = static_cast<uint64_t>(threshold);
    }

    CHECK_ARGUMENT(this->route_max_degree >= 4, "route_max_degree must be at least 4");
    CHECK_ARGUMENT(this->route_ef_construction >= this->route_max_degree,
                   "route_ef_construction must be no less than route_max_degree");

    this->gnoimi_param = std::make_shared<GNOIMIParameter>();
    if (this->partition_strategy_type == IVFPartitionStrategyType::GNO_IMI) {
        CHECK_ARGUMENT(
            json.Contains(IVF_PARTITION_STRATEGY_TYPE_GNO_IMI),
            fmt::format("partition strategy parameters must contains {} when strategy type is {}",
                        IVF_PARTITION_STRATEGY_TYPE_GNO_IMI,
                        IVF_PARTITION_STRATEGY_TYPE_GNO_IMI));
        this->gnoimi_param->FromJson(json[IVF_PARTITION_STRATEGY_TYPE_GNO_IMI]);
    }
}

JsonType
IVFPartitionStrategyParameters::ToJson() const {
    JsonType json;
    if (this->partition_train_type == IVFNearestPartitionTrainerType::KMeansTrainer) {
        json[IVF_TRAIN_TYPE_KEY].SetString(IVF_TRAIN_TYPE_KMEANS);
    } else if (this->partition_train_type == IVFNearestPartitionTrainerType::RandomTrainer) {
        json[IVF_TRAIN_TYPE_KEY].SetString(IVF_TRAIN_TYPE_RANDOM);
    }

    if (this->partition_strategy_type == IVFPartitionStrategyType::IVF) {
        json[IVF_PARTITION_STRATEGY_TYPE_KEY].SetString(IVF_PARTITION_STRATEGY_TYPE_NEAREST);
    } else if (this->partition_strategy_type == IVFPartitionStrategyType::GNO_IMI) {
        json[IVF_PARTITION_STRATEGY_TYPE_KEY].SetString(IVF_PARTITION_STRATEGY_TYPE_GNO_IMI);
    }
    json[IVF_ROUTE_MAX_DEGREE_KEY].SetInt(this->route_max_degree);
    json[IVF_ROUTE_EF_CONSTRUCTION_KEY].SetInt(this->route_ef_construction);
    json[IVF_ENABLE_GPU_BUILD_KEY].SetBool(this->enable_gpu_build);
    json[IVF_GPU_DEVICE_ID_KEY].SetInt(this->gpu_device_id);
    json[IVF_GPU_MEMORY_BUDGET_KEY].SetInt(static_cast<int64_t>(this->gpu_memory_budget));
    json[IVF_GPU_MIN_WORK_THRESHOLD_KEY].SetInt(
        static_cast<int64_t>(this->gpu_min_work_threshold));
    if (this->partition_strategy_type == IVFPartitionStrategyType::GNO_IMI) {
        json[IVF_PARTITION_STRATEGY_TYPE_GNO_IMI].SetJson(this->gnoimi_param->ToJson());
    }
    return json;
}

bool
IVFPartitionStrategyParameters::CheckCompatibility(const ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(IVFPartitionStrategyParameters, p, other);
    CHECK_FIELD_EQ(*this, *p, partition_strategy_type);
    CHECK_FIELD_EQ(*this, *p, route_max_degree);
    CHECK_FIELD_EQ(*this, *p, route_ef_construction);
    CHECK_SUB_PARAM(*this, *p, gnoimi_param);
    return true;
}

}  // namespace vsag
