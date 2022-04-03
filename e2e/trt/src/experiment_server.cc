#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <vector>

#include "experiment_server.h"

std::vector<float> ExperimentServer::RunInferenceOnFiles(const std::vector<std::string>& kFileNames) {
  std::vector<float> output, batch;
  output.reserve(kFileNames.size() * kOutputSingle_);
  batch.reserve(kBatchSize_ * kImSize_);

  /*#pragma omp parallel for
  for (size_t i = 0; i < kFileNames.size(); i++) {
    kLoader_.LoadAndPreproc(kFileNames[i], batch.data());
  }*/

  /*for (size_t i = 0; i < kFileNames.size(); i += kBatchSize_) {
    #pragma omp parallel for
    for (size_t j = 0; j < kBatchSize_; j++) {
      if (i + j < kFileNames.size())
        kLoader_.LoadAndPreproc(kFileNames[i + j]);
    }
  }*/

  /*std::vector<float> batch(kImSize_ * kBatchSize_);
  for (size_t i = 0; i < kFileNames.size(); i += kBatchSize_) {
    #pragma omp parallel for
    for (size_t j = 0; j < kBatchSize_; j++) {
      if (i + j < kFileNames.size()) {
        float *output_buf = batch.data() + j * kImSize;
        kLoader_.LoadAndPreproc(kFileNames[i + j], output_buf);
      }
    }
    // kInfer_->RunInference(batch, kBatchSize_);
    // std::cout << i << std::endl;
  }*/

  #pragma omp parallel for
  for (size_t i = 0; i < kFileNames.size(); i += kBatchSize_) {
    Batch batch;
    batch_queue_.blockingRead(batch);
    for (size_t j = 0; j < kBatchSize_; j++) {
      if (i + j < kFileNames.size()) {
        kLoader_.LoadAndPreproc(
            kFileNames[i + j],
            batch.get()->data() + j * kImSize_);
      }
    }
    const size_t kOutputSize =
        std::min(kFileNames.size() - i, kBatchSize_) * kOutputSingle_;
    kInfer_->RunInference(
        std::make_tuple(std::move(batch), kBatchSize_,
                        output.data() + i * kOutputSingle_, kOutputSize,
                        &batch_queue_));
  }

  kInfer_->Sync();
  return output;
}

float ExperimentServer::TimeEndToEnd(const std::vector<std::string>& kFileNames) {
  auto start = std::chrono::high_resolution_clock::now();
  RunInferenceOnFiles(kFileNames);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  return diff.count() / 1000.0;
}

void ExperimentServer::RunInferenceOnCompressed(
    const std::vector<CompressedImage>& kCompressedImages,
    std::vector<float> *output) {
  #pragma omp parallel for
  for (size_t i = 0; i < kCompressedImages.size(); i += kBatchSize_) {
    Batch batch;
    batch_queue_.blockingRead(batch);
    for (size_t j = 0; j < kBatchSize_; j++) {
      if (i + j < kCompressedImages.size()) {
        kLoader_.DecodeAndPreproc(
            kCompressedImages[i + j],
            batch.get()->data() + j * kImSize_);
      }
    }
    if (kRunInfer_) {
      const size_t kOutputSize =
          std::min(kCompressedImages.size() - i, kBatchSize_) * kOutputSingle_;
      kInfer_->RunInference(
          std::make_tuple(std::move(batch), kBatchSize_,
                          output->data() + i * kOutputSingle_,
                          kOutputSize,
                          &batch_queue_));
    } else {
      batch_queue_.blockingWrite(std::move(batch));
    }
  }
  /*std::vector<std::future<void> > async_results;
  for (size_t i = 0; i < kCompressedImages.size(); i += kBatchSize_) {
    async_results.push_back(std::async(
            std::launch::async,
            [this](const std::vector<CompressedImage>& kCompressedImages,
                   std::vector<float> *output,
                   const size_t i) {
              Batch batch;
              batch_queue_.blockingRead(batch);
              for (size_t j = 0; j < kBatchSize_; j++) {
                if (i + j < kCompressedImages.size()) {
                  kLoader_.DecodeAndPreproc(
                      kCompressedImages[i + j],
                      batch.get()->data() + j * kImSize_);
                }
              }
              kInfer_->RunInference(
                  std::make_tuple(std::move(batch), kBatchSize_,
                                  output->data() + i * kOutputSingle_,
                                  kBatchSize_ * kOutputSingle_,
                                  &batch_queue_));
            },
            kCompressedImages, output, i));
  }
  for (size_t i = 0; i < async_results.size(); i++)
    async_results[i].get();*/

  /*for (size_t i = 0; i < kCompressedImages.size(); i += kBatchSize_) {
    #pragma omp parallel for
    for (size_t j = 0; j < kBatchSize_; j++) {
      if (i + j < kCompressedImages.size()) {
        float *output_buf = batch.data() + j * kImSize;
        kLoader_.DecodeAndPreproc(kCompressedImages[i + j], output_buf);
      }
    }
    // kInfer_->RunInference(batch, kBatchSize_);
    // std::cout << i << std::endl;
  }*/
  kInfer_->Sync();
}



std::pair<float, std::vector<float> > ExperimentServer::TimeNoLoad(const std::vector<CompressedImage>& kCompressedImages) {
  std::vector<float> output(kCompressedImages.size() * kOutputSingle_);

  auto start = std::chrono::high_resolution_clock::now();
  RunInferenceOnCompressed(kCompressedImages, &output);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  float time = diff.count() / 1000.0;
  return std::make_pair(time, output);
}

float ExperimentServer::TimeInferenceOnly() {
  const size_t kNbBatches = 1000;
  // BatchBase output(kNbBatches * kBatchSize_ * kOutputSingle_);
  std::vector<float> output(kNbBatches * kBatchSize_ * kOutputSingle_);

  auto start = std::chrono::high_resolution_clock::now();

  #pragma omp parallel for
  for (size_t i = 0; i < kNbBatches; i++) {
    Batch batch;
    batch_queue_.blockingRead(batch);
    kInfer_->RunInference(
        std::make_tuple(
            std::move(batch), kBatchSize_,
            output.data() + i * kBatchSize_ * kOutputSingle_,
            kBatchSize_ * kOutputSingle_,
            &batch_queue_));
  }
  kInfer_->Sync();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  return diff.count() / 1000.0;
}

float ExperimentServer::TimeDecodePreprocOnly(const std::vector<CompressedImage>& kCompressedImages) {
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<float> batch;
  batch.reserve(kBatchSize_ * kImSize_);

  /*#pragma omp parallel for
  for (size_t i = 0; i < kCompressedImages.size(); i++) {
    size_t offset = i % kBatchSize_;
    kLoader_.DecodeAndPreproc(
        kCompressedImages[i],
        batch.data() + offset * kImSize_);
  }*/
  #pragma omp parallel for
  for (size_t i = 0; i < kCompressedImages.size(); i += kBatchSize_) {
    for (size_t j = 0; j < kBatchSize_; j++) {
      if (i + j < kCompressedImages.size()) {
        kLoader_.DecodeAndPreproc(
            kCompressedImages[i + j],
            batch.data() + j * kImSize_);
      }
    }
  }


  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  return diff.count() / 1000.0;
}
