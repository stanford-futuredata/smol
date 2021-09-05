#ifndef DATA_LOADER_H_
#define DATA_LOADER_H_

#include <memory>
#include <stdint.h>
#include <string>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "common.h"


class DataLoader {
 protected:
  const size_t kResizeDim_;
  const size_t kModelInputDim_;
  const bool kDoResize_;
  const LoaderCondition kCondition_;

 public:
  DataLoader(const size_t kResizeDim, const size_t kModelInputDim,
             const bool kDoResize, const LoaderCondition cond) :
      kResizeDim_(kResizeDim), kModelInputDim_(kModelInputDim),
      kDoResize_(kDoResize), kCondition_(cond) {}
  ~DataLoader() {}

  size_t GetResol() const { return kModelInputDim_; }

  CompressedImage LoadCompressedImageFromFile(const std::string& kFileName) const;
  virtual cv::Mat DecodeImage(CompressedImage kCompressedBuf) const = 0;
  virtual void PreprocessImage(const cv::Mat& kRawImage, float *output_buf) const = 0;

  void DecodeAndPreproc(CompressedImage kCompressed, float *output_buf) const;
  void LoadAndPreproc(const std::string& kFileName, float *output_buf) const;
};


class NaiveDataLoader : public DataLoader {
 public:
  using DataLoader::DataLoader;

  cv::Mat DecodeImage(CompressedImage kCompressedBuf) const;
  void PreprocessImage(const cv::Mat& kRawImage, float *output_buf) const;
};


class OptimizedDataLoader : public DataLoader {
 private:
  float map_[3][256];

 public:
  OptimizedDataLoader(const size_t kResizeDim, const size_t kModelInputDim,
                      const bool kDoResize, LoaderCondition cond) :
      DataLoader(kResizeDim, kModelInputDim, kDoResize, cond) {
    float means[3] = {0.485, 0.456, 0.406};
    float stds[3] = {0.229, 0.224, 0.225};
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 256; j++) {
        map_[i][j] = (j / 255.0 - means[i]) / stds[i];
      }
    }
  }

  cv::Mat DecodeImage(CompressedImage kCompressedBuf) const;
  void PreprocessImage(const cv::Mat& kRawImage, float *output_buf) const;
};


class PNGDataLoader : public DataLoader {
 public:
  using DataLoader::DataLoader;

  cv::Mat DecodeImage(CompressedImage kCompressedBuf) const;
  void PreprocessImage(const cv::Mat& kRawImage, float *output_buf) const;
};

class OptPNGDataLoader : public DataLoader {
 protected:
  float map_[3][256];

 public:
  OptPNGDataLoader(const size_t kResizeDim, const size_t kModelInputDim,
                   const bool kDoResize, LoaderCondition cond) :
      DataLoader(kResizeDim, kModelInputDim, kDoResize, cond) {
    assert(kDoResize_ == true);
    float means[3] = {0.485, 0.456, 0.406};
    float stds[3] = {0.229, 0.224, 0.225};
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 256; j++) {
        map_[i][j] = (j / 255.0 - means[i]) / stds[i];
      }
    }
  }

  cv::Mat DecodeImage(CompressedImage kCompressedBuf) const;
  void PreprocessImage(const cv::Mat& kRawImage, float *output_buf) const;
};

class OptResizePNGDataLoader : public OptPNGDataLoader {
 public:
  using OptPNGDataLoader::OptPNGDataLoader;

  void PreprocessImage(const cv::Mat& kRawImage, float *output_buf) const;
};


// TODO: probably should return cv::Size for named
static std::pair<size_t, size_t> RatioPreservingResize(
    const size_t kResizeDim, const size_t kOrigWidth, const size_t kOrigHeight) {
  if (kOrigWidth > kOrigHeight) {
    float adj_width = kResizeDim * (kOrigWidth / (float) kOrigHeight);
    return std::make_pair((size_t) adj_width, kResizeDim);
  } else {
    float adj_height = kResizeDim * (kOrigHeight / (float) kOrigWidth);
    return std::make_pair(kResizeDim, (size_t) adj_height);
  }
}

#endif // DATA_LOADER_H_
