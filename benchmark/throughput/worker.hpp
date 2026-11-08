
#pragma once

#include <cstddef>

constexpr std::size_t NDIM = 3;
constexpr int BLOCKSIZE = 512;

// The interface must not mention clue:: types: clue::PointsHost lives in a backend-specific inline namespace
// (CLUE_ABI_TAG), so it is a different type in main.cpp (serial backend) and in the CUDA worker. Each backend
// therefore owns its events (EventStore) and main.cpp only passes opaque pointers and event indices.
namespace serial {
  struct WorkerState;
  struct EventStore;
  WorkerState* createWorker(float dc, float rhoc, float outlier, int n_points);
  void destroyWorker(WorkerState*);
  EventStore* createEvents(int nEvents, int nPoints, int nClusters);
  void destroyEvents(EventStore*);
  void processEvent(WorkerState*, EventStore*, int eventId);
}  // namespace serial

namespace cuda {
  struct WorkerState;
  struct EventStore;
  WorkerState* createWorker(float dc, float rhoc, float outlier, int n_points);
  void destroyWorker(WorkerState*);
  EventStore* createEvents(int nEvents, int nPoints, int nClusters);
  void destroyEvents(EventStore*);
  void processEvent(WorkerState*, EventStore*, int eventId);
}  // namespace cuda
