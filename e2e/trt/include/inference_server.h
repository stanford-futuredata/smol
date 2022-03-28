#ifndef INFERENCE_SERVER_H_
#define INFERENCE_SERVER_H_

#include <memory>
#include <stdint.h>
#include <string>
#include <thread>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "folly/MPMCQueue.h"

#include "cuda_wrapper.h"
#include "io_helper.h"

#include "common.h"
#include "data_loader.h"
#include "calibrator.h"


// This should possibly be an abstract base class, but we're only using ONNX for now
class InferenceServer {
 public:
  virtual size_t GetOutputSingle() = 0;
  virtual void RunInference(QueueData data) = 0;
  virtual void Sync() = 0;
  virtual std::vector<std::vector<float> > GetResults() = 0;
};

class OnnxInferenceServer : public InferenceServer {
 private:
  const size_t kBatchSize_;
  size_t kOutputSingle_; // Should be const, but lazy
  const size_t MAX_WORKSPACE_SIZE = 1ULL << 30;
  const static size_t kNbStreams_ = 16;
  const bool kDoMemcpy_;

  nvinfer1::Logger gLogger;
  std::unique_ptr<nvinfer1::ICudaEngine> engine{nullptr};
  // std::unique_ptr<nvinfer1::IExecutionContext, nvinfer1::Destroy<nvinfer1::IExecutionContext>> context{nullptr};
  std::vector<std::unique_ptr<nvinfer1::IExecutionContext> > contexts;
  void *bindings[kNbStreams_][2];
  cudawrapper::CudaStream streams[kNbStreams_];

  folly::MPMCQueue<QueueData> queue_;
  std::vector<std::thread> threads_;


  void LoadAndLaunch();

  nvinfer1::ICudaEngine* CreateCudaEngine(
      const std::string& kOnnxPath, const std::string& kOnnxPathBS1,
      BaseCalibrator *calibrator,
      const bool kDoINT8, const bool kAddResize);
  nvinfer1::ICudaEngine* GetCudaEngine(const std::string& kEnginePath);

  void _RunInferenceThread(const size_t idx);

  void teardown() {
    for (size_t i = 0; i < kNbStreams_; i++) {
      queue_.blockingWrite(std::make_tuple(nullptr, 0, nullptr, 0, nullptr));
    }
    for (size_t i = 0; i < threads_.size(); i++)
      threads_[i].join();
    Sync();

    for (size_t i = 0; i < kNbStreams_; i++)
      for (void* ptr : this->bindings[i])
        cudaFree(ptr);
  }



 public:
  OnnxInferenceServer(
      const std::string& kEnginePath, const size_t kBatchSize,
      const bool kDoMemcpy);

  OnnxInferenceServer(
      const std::string& kOnnxPath, const std::string& kOnnxPathBS1,
      const std::string& kCachePath,
      const size_t kBatchSize, const bool kDoMemcpy,
      const DataLoader *kLoader = nullptr,
      const std::vector<CompressedImage>& kCompressedImages = std::vector<CompressedImage>(0),
      const bool kDoINT8 = false,
      const bool kAddResize = false);

  OnnxInferenceServer(
      const std::string& kOnnxPath, const std::string& kOnnxPathBS1,
      const std::string& kCachePath,
      const size_t kBatchSize, const bool kDoMemcpy,
      const VideoDataLoader *kLoader = nullptr,
      const std::vector<std::string>& kFileName = std::vector<std::string>(0),
      const bool kDoINT8 = false,
      const bool kAddResize = false);

  size_t GetOutputSingle() { return kOutputSingle_; }

  void warmup(const size_t kResol);

  void RunInference(QueueData data);

  void Sync();

  std::vector<std::vector<float> > GetResults();

  ~OnnxInferenceServer() {
    teardown();
  }
};

#endif // INFERENCE_SERVER_H_
