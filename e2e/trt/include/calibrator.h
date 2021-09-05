#pragma once

#include <fstream>
#include <iostream>
#include <iterator>

#include <NvInfer.h>

#include "common.h"
#include "data_loader.h"
#include "video_data_loader.h"

class BaseCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
 protected:
  const size_t kBatchSize_;
  const size_t kImSize_;
  const size_t kInputSize_;
  size_t counter_;
  BatchBase inp_data_;
  void *device_ptr_;

  std::vector<char> calibration_cache_;
  bool read_cache_ = false;

  static std::string calibrationTableName() {
    return std::string("CalibrationTable"); // FIXME
  }

 public:
  BaseCalibrator(const size_t kBatchSize, const size_t kImSize) :
      kBatchSize_(kBatchSize),
      kImSize_(kImSize),
      kInputSize_(kBatchSize_ * kImSize_),
      inp_data_(kInputSize_) {
    counter_ = 0;
    cudaMalloc(&device_ptr_, kInputSize_ * sizeof(float));
  }

  virtual ~BaseCalibrator() {
    cudaFree(device_ptr_);
  }

  int getBatchSize() const override {
    return 1;  // FIXME: ONNX hack, fix once TensorRT fixes the bug
  }

  virtual void fillInpData() = 0;
  virtual size_t getSize() = 0;

  bool getBatch(void* bindings[], const char* names[], int nbBindings) override {
    if (counter_ / kBatchSize_ > 500 || counter_ + kBatchSize_ > getSize()) {
      counter_ = 0;
      return false;
    }

    fillInpData();

    cudaMemcpy(device_ptr_, inp_data_.data(), kInputSize_ * sizeof(float),
               cudaMemcpyHostToDevice);
    bindings[0] = device_ptr_;
    counter_ += kBatchSize_;

    return true;
  }

  const void* readCalibrationCache(size_t& length) override {
    calibration_cache_.clear();
    std::ifstream input(calibrationTableName(), std::ios::binary);
    input >> std::noskipws;
    if (read_cache_ && input.good()) {
      std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                std::back_inserter(calibration_cache_));
    }

    length = calibration_cache_.size();
    return length ? &calibration_cache_[0] : nullptr;
  }

  void writeCalibrationCache(const void* cache, size_t length) override {
    std::ofstream output(calibrationTableName(), std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
  }
};


class ImageCalibrator : public BaseCalibrator {
 private:
  const DataLoader *kLoader_;
  const std::vector<CompressedImage>& kCompressedImages_;

 public:
  ImageCalibrator(const DataLoader *kLoader,
                  const std::vector<CompressedImage>& kCompressedImages,
                  const size_t kBatchSize) :
      kLoader_(kLoader), kCompressedImages_(kCompressedImages),
      BaseCalibrator(kBatchSize, 3 * kLoader->GetResol() * kLoader->GetResol()) {}

  void fillInpData() {
    #pragma omp parallel for
    for (size_t j = 0; j < kBatchSize_; j++) {
      size_t offset = counter_ + j;
      kLoader_->DecodeAndPreproc(
          kCompressedImages_[offset], inp_data_.data() + j * kImSize_);
    }
  }

  size_t getSize() {
    return kCompressedImages_.size();
  }
};


class VideoCalibrator : public BaseCalibrator {
 private:
  const VideoDataLoader *kLoader_;
  const std::vector<std::string>& kFileNames_;

 public:
  VideoCalibrator(const VideoDataLoader *kLoader,
                  const std::vector<std::string>& kFileNames,
                  const size_t kBatchSize) :
      kLoader_(kLoader), kFileNames_(kFileNames),
      // hack due to Image
      BaseCalibrator(1, 3 * kBatchSize * kLoader->GetResol() * kLoader->GetResol()) {}

  void fillInpData() {
    kLoader_->DecodeAndPreprocessGOP(kFileNames_[counter_], inp_data_.data());
  }

  size_t getSize() {
    return kFileNames_.size();
  }
};
