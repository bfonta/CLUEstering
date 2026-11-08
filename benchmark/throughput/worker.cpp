
#include "worker.hpp"
#include "defines.hpp"
#include "utils/generation.hpp"

#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"

#include <alpaka/alpaka.hpp>

#include <utility>
#include <vector>

namespace backend {

  using Acc = ALPAKA_BACKEND::Acc1D;
  using Device = ALPAKA_BACKEND::Device;
  using Queue = ALPAKA_BACKEND::Queue;

  struct WorkerState {
    Device device;
    Queue queue;
    clue::Clusterer<NDIM> clusterer;
    clue::PointsDevice<NDIM> d_points;

    WorkerState(float dc, float rhoc, float outlier, int n_points)
        : device(alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0u)),
          queue(device),
          clusterer(queue, dc, rhoc, outlier),
          d_points(queue, n_points) {}
  };

  WorkerState* createWorker(float dc, float rhoc, float outlier, int n_points) {
    return new WorkerState(dc, rhoc, outlier, n_points);
  }

  void destroyWorker(WorkerState* w) { delete w; }

  struct EventStore {
    ::std::vector<clue::PointsHost<NDIM>> events;
  };

  EventStore* createEvents(int nEvents, int nPoints, int nClusters) {
    const auto device = alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0u);
    Queue queue(device);
    auto* store = new EventStore;
    store->events.reserve(nEvents);
    for (auto i = 0; i < nEvents; ++i) {
      store->events.emplace_back(clue::utils::generateRandomData<NDIM>(
          queue, nPoints, nClusters, ::std::make_pair(-100.f, 100.f), 1.f, 0.1f, i));
    }
    return store;
  }

  void destroyEvents(EventStore* s) { delete s; }

  void processEvent(WorkerState* w, EventStore* s, int eventId) {
    w->clusterer.make_clusters(w->queue, s->events[eventId], w->d_points);
    alpaka::wait(w->queue);
  }

}  // namespace backend
