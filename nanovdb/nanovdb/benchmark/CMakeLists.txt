# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
#[=======================================================================[

  CMake Configuration for NanoVDB Benchmarks

#]=======================================================================]

cmake_minimum_required(VERSION 3.24)
project(NanoVDBBenchmarks LANGUAGES CXX)

if(NOT NANOVDB_USE_CUDA)
  message(WARNING " - CUDA required to build NanoVDB benchmarks. Skipping.")
  return()
endif()

add_executable(benchmark_distributed_points_to_grid
  DistributedPointsToGrid.cu)

target_link_libraries(benchmark_distributed_points_to_grid PRIVATE nanovdb)

if(VDB_MSVC_RUNTIME_SELECTION)
  set_target_properties(benchmark_distributed_points_to_grid PROPERTIES
    MSVC_RUNTIME_LIBRARY ${VDB_MSVC_RUNTIME_SELECTION})
endif()
