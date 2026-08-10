// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/cuda/DeviceMesh.h>
#include <nanovdb/tools/cuda/DistributedPointsToGrid.cuh>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {

constexpr int kWarmupCount = 10;
constexpr int kRunCount = 20;
constexpr size_t kDefaultPointCount = size_t(1) << 20;
constexpr int32_t kMajorRadius = 100;
constexpr int32_t kMinorRadius = 50;

/// Return uniformly distributed points within a solid torus in the xy-plane.
void sampleTorus(nanovdb::Coord *points, size_t pointCount) {
  std::mt19937 generator(42u);
  std::uniform_int_distribution<int32_t> xyDistribution(
      -(kMajorRadius + kMinorRadius), kMajorRadius + kMinorRadius);
  std::uniform_int_distribution<int32_t> zDistribution(-kMinorRadius,
                                                       kMinorRadius);

  const int32_t minorRadiusSquared = kMinorRadius * kMinorRadius;
  for (size_t i = 0; i < pointCount;) {
    const int32_t x = xyDistribution(generator);
    const int32_t y = xyDistribution(generator);
    const int32_t z = zDistribution(generator);
    const double radialDistance = std::sqrt(double(x * x + y * y));
    const double tubeDistance = radialDistance - kMajorRadius;
    if (tubeDistance * tubeDistance + z * z <= minorRadiusSquared) {
      points[i++] = nanovdb::Coord(x, y, z);
    }
  }
}

void synchronize(const nanovdb::cuda::DeviceMesh &deviceMesh) {
  for (const auto &[deviceId, stream] : deviceMesh) {
    cudaCheck(cudaSetDevice(deviceId));
    cudaCheck(cudaStreamSynchronize(stream));
  }
}

size_t parsePointCount(int argc, char *argv[]) {
  if (argc > 2) {
    throw std::runtime_error(
        "Usage: benchmark_distributed_points_to_grid [point_count]");
  }
  if (argc == 1)
    return kDefaultPointCount;

  const std::string argument(argv[1]);
  if (argument.empty() || argument.front() == '-') {
    throw std::runtime_error("point_count must be a positive integer");
  }

  size_t parsedCharacters = 0;
  const size_t pointCount = std::stoull(argument, &parsedCharacters);
  if (parsedCharacters != argument.size() || pointCount == 0) {
    throw std::runtime_error("point_count must be a positive integer");
  }
  return pointCount;
}

} // namespace

int main(int argc, char *argv[]) try {
  const size_t pointCount = parsePointCount(argc, argv);

  int cudaDeviceCount = 0;
  const cudaError_t deviceCountStatus = cudaGetDeviceCount(&cudaDeviceCount);
  if (deviceCountStatus == cudaErrorNoDevice || cudaDeviceCount == 0) {
    throw std::runtime_error("No CUDA devices are available");
  }
  cudaCheck(deviceCountStatus);
  nanovdb::cuda::DeviceMesh deviceMesh;

  nanovdb::Coord *points = nullptr;
  cudaCheck(cudaMallocManaged(&points, pointCount * sizeof(nanovdb::Coord)));
  sampleTorus(points, pointCount);

  using BuildT = nanovdb::ValueOnIndex;
  nanovdb::tools::cuda::DistributedPointsToGrid<BuildT> converter(deviceMesh);

  std::cout << "DistributedPointsToGrid benchmark\n"
            << "  CUDA devices: " << deviceMesh.deviceCount() << '\n'
            << "  Points: " << pointCount << '\n'
            << "  Warmups: " << kWarmupCount << '\n'
            << "  Measured runs: " << kRunCount << std::endl;

  for (int run = 0; run < kWarmupCount; ++run) {
    auto handle = converter.getHandle(points, pointCount);
    synchronize(deviceMesh);
  }

  using Clock = std::chrono::steady_clock;
  std::chrono::duration<double, std::milli> totalTime{0.0};
  for (int run = 0; run < kRunCount; ++run) {
    const auto start = Clock::now();
    auto handle = converter.getHandle(points, pointCount);
    synchronize(deviceMesh);
    totalTime += Clock::now() - start;
  }

  const double averageMilliseconds = totalTime.count() / kRunCount;
  const double millionPointsPerSecond =
      static_cast<double>(pointCount) / (averageMilliseconds * 1000.0);

  std::cout << std::fixed << std::setprecision(3)
            << "Average construction time: " << averageMilliseconds << " ms\n"
            << "Throughput: " << millionPointsPerSecond << " million points/s"
            << std::endl;

  cudaCheck(cudaFree(points));
  return 0;
} catch (const std::exception &error) {
  std::cerr << "Error: " << error.what() << std::endl;
  return 1;
}
