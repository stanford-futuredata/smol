#ifndef VIDEO_EXPERIMENT_SERVER_H_
#define VIDEO_EXPERIMENT_SERVER_H_

#include "folly/MPMCQueue.h"
#include "omp.h"

#include "video_data_loader.h"
#include "inference_server.h"
#include "common.h"

class VideoExperimentServer {
 private:
  const VideoDataLoader& kLoader_;
  InferenceServer *kInfer_; // not const cause this would be a pain
  const size_t kBatchSize_;
  const size_t kImSize_;
  const size_t kOutputSingle_;
  folly::MPMCQueue<Batch> batch_queue_;

  const bool kRunInfer_;

 public:
  VideoExperimentServer(const VideoDataLoader& kLoader, InferenceServer *kInfer,
                        const size_t kBatchSize, const bool kRunInfer) :
      kLoader_(kLoader), kInfer_(kInfer), kBatchSize_(kBatchSize),
      kImSize_(3 * kLoader.GetResol() * kLoader.GetResol()),
      kOutputSingle_(kInfer->GetOutputSingle()),
      batch_queue_(omp_get_max_threads() * 3),
      kRunInfer_(kRunInfer) {
    for (size_t i = 0; i < omp_get_max_threads() * 3; i++) {
      batch_queue_.blockingWrite(
          std::make_unique<BatchBase>(kBatchSize_ * kImSize_));
    }
  }

  void RunInferenceOnFiles(
      const std::vector<std::string>& kFileNames,
      std::vector<float> *output);
  std::pair<float, std::vector<float> > TimeEndToEnd(const std::vector<std::string>& kFileNames);
};

#endif // VIDEO_EXPERIMENT_SERVER_H_
