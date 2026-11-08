
#include "worker.hpp"


#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

constexpr auto dc = 1.5f, rhoc = 10.f, outlier = 1.5f;

auto runEvents(int nCpuThreads, int nGpuStreams, int nEvents, int nPoints, int nClusters) {
  // events are generated once, before the clock starts, by the backends that have workers
  auto* cpuEvents = nCpuThreads > 0 ? serial::createEvents(nEvents, nPoints, nClusters) : nullptr;
#ifdef CLUE_HAS_CUDA
  auto* gpuEvents = nGpuStreams > 0 ? cuda::createEvents(nEvents, nPoints, nClusters) : nullptr;
#endif

  std::vector<serial::WorkerState*> cpuWorkers(nCpuThreads);
  for (auto& w : cpuWorkers)
    w = serial::createWorker(dc, rhoc, outlier, nPoints);

#ifdef CLUE_HAS_CUDA
  std::vector<cuda::WorkerState*> gpuWorkers(nGpuStreams);
  for (auto& w : gpuWorkers)
    w = cuda::createWorker(dc, rhoc, outlier, nPoints);
#else
  if (nGpuStreams > 0) {
    std::cerr << "GPU support not available in this build\n";
    return 0.;
  }
#endif

  std::atomic<int> eventCounter = 0;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread cpuThread([&] {
    if (nCpuThreads == 0)
      return;
    tbb::task_arena arena(nCpuThreads);
    arena.execute([&] {
      tbb::parallel_for(0, nCpuThreads, [&](int i) {
        while (true) {
          int id = eventCounter.fetch_add(1);
          if (id >= nEvents)
            return;
          serial::processEvent(cpuWorkers[i], cpuEvents, id);
        }
      });
    });
  });

#ifdef CLUE_HAS_CUDA
  std::thread gpuThread([&] {
    if (nGpuStreams == 0)
      return;
    tbb::task_arena arena(nGpuStreams);
    arena.execute([&] {
      tbb::parallel_for(0, nGpuStreams, [&](int i) {
        while (true) {
          int id = eventCounter.fetch_add(1);
          if (id >= nEvents)
            return;
          cuda::processEvent(gpuWorkers[i], gpuEvents, id);
        }
      });
    });
  });
#endif

  cpuThread.join();
#ifdef CLUE_HAS_CUDA
  gpuThread.join();
#endif

  auto end = std::chrono::high_resolution_clock::now();

  for (auto& w : cpuWorkers)
    serial::destroyWorker(w);
  if (cpuEvents)
    serial::destroyEvents(cpuEvents);
#ifdef CLUE_HAS_CUDA
  for (auto& w : gpuWorkers)
    cuda::destroyWorker(w);
  if (gpuEvents)
    cuda::destroyEvents(gpuEvents);
#endif

  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  return (1000. * nEvents) / duration;
}

int main(int argc, char* argv[]) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <nCpuThreads> <nGpuStreams> <nEvents> <nPoints> <nClusters>\n";
    return 1;
  }
  const auto nCpuThreads = std::stoi(argv[1]);
  const auto nGpuStreams = std::stoi(argv[2]);
  const auto nEvents = std::stoi(argv[3]);
  const auto nPoints = std::stoi(argv[4]);
  const auto nClusters = std::stoi(argv[5]);

  const auto throughput = runEvents(nCpuThreads, nGpuStreams, nEvents, nPoints, nClusters);
  std::cout << "Throughput = " << throughput << " events/s\n";
}
