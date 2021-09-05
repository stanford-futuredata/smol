#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <vector>

#include "video_experiment_server.h"

// void VideoExperimentServer::RunInferenceOnFiles(
//     const std::vector<std::string>& kFileNames,
//     std::vector<float> *output) {
//   #pragma omp parallel for
//   for (size_t i = 0; i < kFileNames.size(); i++) {
//     QueueData input_data;
// 
//     std::cerr << "Processing " << i << std::endl;
//     Batch batch;
//     batch_queue_.blockingRead(batch);
//     kLoader_.DecodeAndPreprocessGOP(kFileNames[i], batch.get()->data());
// 
//     batch_queue_.blockingWrite(std::move(batch));
//     /*kInfer_->RunInference(
//         std::make_tuple(std::move(batch), kBatchSize_,
//                         output->data() + i * kOutputSingle_, kBatchSize_ * kOutputSingle_,
//                         &batch_queue_));*/
//   }
//   kInfer_->Sync();
// }


void VideoExperimentServer::RunInferenceOnFiles(
    const std::vector<std::string>& kFileNames,
    std::vector<float> *output) {

  /*std::vector<std::future<void> > async_results;
  for (size_t i = 0; i < kFileNames.size(); i++) {
    async_results.push_back(std::async(
        std::launch::async,
        [this](const size_t i, const std::string& kFileName,
               std::vector<float> *output,
               folly::MPMCQueue<Batch> *batch_queue) {
          Batch batch;
          batch_queue->blockingRead(batch);
          kLoader_.DecodeAndPreprocessGOP(kFileName, batch.get()->data());

          // batch_queue->blockingWrite(std::move(batch));
          kInfer_->RunInference(
              std::make_tuple(std::move(batch), kBatchSize_,
                              output->data() + i * kBatchSize_ * kOutputSingle_,
                              kBatchSize_ * kOutputSingle_,
                              &batch_queue_));
        },
        i, kFileNames[i], output, &batch_queue_));
    // async_results.push_back(f);
  }
  for (size_t i = 0; i < async_results.size(); i++)
    async_results[i].get();*/
  #pragma omp parallel for
  for (size_t i = 0; i < kFileNames.size(); i++) {
    Batch batch;
    batch_queue_.blockingRead(batch);

    kLoader_.DecodeAndPreprocessGOP(kFileNames[i], batch.get()->data());
    if (kRunInfer_) {
      kInfer_->RunInference(
          std::make_tuple(
              std::move(batch), kBatchSize_,
              output->data() + i * kBatchSize_ * kOutputSingle_,
              kBatchSize_ * kOutputSingle_,
              &batch_queue_));
    } else {
      batch_queue_.blockingWrite(std::move(batch));
    }
  }

  kInfer_->Sync();
}

std::pair<float, std::vector<float> > VideoExperimentServer::TimeEndToEnd(const std::vector<std::string>& kFileNames) {
  std::vector<float> output(kFileNames.size() * kOutputSingle_ * kBatchSize_);

  auto start = std::chrono::high_resolution_clock::now();
  RunInferenceOnFiles(kFileNames, &output);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  return std::make_pair(diff.count() / 1000.0, output);
}
