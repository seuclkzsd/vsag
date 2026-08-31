# Copyright 2024-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# VSAG installs its C/C++ warning and ISA flags with add_compile_options(), so
# they are inherited by every language in this directory -- including CUDA,
# where nvcc rejects them (e.g. "-Werror -fno-omit-frame-pointer" parses as
# -Werror=<value>). Clear the inherited list here and give the CUDA target only
# the flags it actually needs; host-side flags go through -Xcompiler.
set_property (DIRECTORY PROPERTY COMPILE_OPTIONS "")

if (ENABLE_CUDA)
    add_library (vsag_gpu STATIC cuda_kmeans_assign.cu)
    set_target_properties (vsag_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES "${VSAG_CUDA_ARCHITECTURES}")
    target_compile_options (vsag_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>)
    target_compile_definitions (vsag_gpu PUBLIC VSAG_ENABLE_CUDA=1)
    target_link_libraries (vsag_gpu PUBLIC CUDA::cudart CUDA::cublas)
else ()
    add_library (vsag_gpu STATIC cuda_kmeans_assign_stub.cpp)
    set_target_properties (vsag_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # Only the stub is instrumented: it is ordinary C++ that the tests below
    # exercise. The CUDA branch is left out because nvcc does not take
    # --coverage, and a device-less CI cannot run those kernels anyway.
    target_link_libraries (vsag_gpu PRIVATE coverage_config)
endif ()

target_include_directories (vsag_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
maybe_add_dependencies (vsag_gpu spdlog boost mkl openblas)

if (ENABLE_TESTS)
    # gpu_plan.h is deliberately free of CUDA so the sizing decisions can be
    # covered on a machine without a device, which is what CI has. The kernels
    # themselves stay untested here.
    file (GLOB GPU_TESTS "*_test.cpp")
    add_library (gpu_test STATIC ${GPU_TESTS})
    # This directory clears the inherited compile options because nvcc rejects
    # them, so put the project's C++ flags back on the host-only test target.
    separate_arguments (_gpu_test_flags NATIVE_COMMAND "${VSAG_EXTERNAL_CXX_FLAGS}")
    target_compile_options (gpu_test PRIVATE ${_gpu_test_flags})
    target_link_libraries (gpu_test PRIVATE Catch2::Catch2 vsag_static unittest_deps
                                          coverage_config)
    vsag_link_openmp_private (gpu_test)
    add_dependencies (gpu_test Catch2)
endif ()
