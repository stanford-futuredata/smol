#ifndef VIDEO_DATA_LOADER_H_
#define VIDEO_DATA_LOADER_H_

#include <memory>
#include <stdint.h>
#include <string>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "common.h"
#include "video_decoder.h"

class VideoDataLoader {
 protected:
  const size_t kResizeDim_; // WARNING: UNUSED
  const size_t kModelInputDim_;
  const CropRegion kRegion_;
  const LoaderCondition kCondition_;

  void InitDecoder() const;

 public:
  VideoDataLoader(const size_t kResizeDim, const size_t kModelInputDim,
                  const CropRegion region, const LoaderCondition cond) :
      kResizeDim_(kResizeDim), kModelInputDim_(kModelInputDim),
      kRegion_(region), kCondition_(cond) {}
  ~VideoDataLoader() {}

  size_t GetResol() const { return kModelInputDim_; }

  CompressedImage LoadCompressedImageFromFile(const std::string& kFileName) const;

  virtual std::vector<cv::Mat> DecodeGOP(const std::string& kFileName) const = 0;
  virtual void PreprocessGOP(const std::vector<cv::Mat>& kRawGOP, float *output_buf) const = 0;

  virtual void DecodeAndPreprocessGOP(const std::string& kFileName, float *output_buf) const = 0;
};



class OptimizedVidDataLoader : public VideoDataLoader {
 private:
  float map_[3][256];

 public:
  OptimizedVidDataLoader(
      const size_t kResizeDim, const size_t kModelInputDim,
      const CropRegion region, const LoaderCondition cond) :
      VideoDataLoader(kResizeDim, kModelInputDim, region, cond) {
    float means[3] = {0.485, 0.456, 0.406};
    float stds[3] = {0.229, 0.224, 0.225};
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 256; j++) {
        map_[i][j] = (j / 255.0 - means[i]) / stds[i];
      }
    }
  }

  std::vector<cv::Mat> DecodeGOP(const std::string& kFileName) const;
  void PreprocessGOP(const std::vector<cv::Mat>& kRawGOP, float *output_buf) const;

  void DecodeAndPreprocessGOP(const std::string& kFileName, float *output_buf) const;
};

class NaiveVidDataLoader : public VideoDataLoader {
 public:
  using VideoDataLoader::VideoDataLoader;

  std::vector<cv::Mat> DecodeGOP(const std::string& kFileName) const;
  void PreprocessGOP(const std::vector<cv::Mat>& kRawGOP, float *output_buf) const;

  void DecodeAndPreprocessGOP(const std::string& kFileName, float *output_buf) const;
};



#endif // VIDEO_DATA_LOADER_H_
