#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "turbojpeg.h"
#include "jpeglib.h"

#include "data_loader.h"

cv::Mat NaiveDataLoader::DecodeImage(CompressedImage compressed) const {
  cv::Mat raw_data(1, compressed.second, CV_8UC1, (void*) compressed.first);
  cv::Mat decoded_img = cv::imdecode(raw_data, cv::IMREAD_COLOR);
  cv::Mat ret;
  cv::cvtColor(decoded_img, ret, cv::COLOR_BGR2RGB);
  return ret;
}

void NaiveDataLoader::PreprocessImage(const cv::Mat& kRawImage, float *output_buf) const {
  assert(kDoResize_ == true);
  if (kCondition_ == LoaderCondition::DecodeOnly)
    return;
  cv::Mat resized, normalized;

  // Resize
  auto new_resol = RatioPreservingResize(kResizeDim_, kRawImage.cols, kRawImage.rows);
  cv::resize(kRawImage, resized, cv::Size(new_resol.first, new_resol.second));
  if (kCondition_ == LoaderCondition::DecodeResize)
    return;

  // Center crop
  size_t x = (size_t) round((resized.cols - kModelInputDim_) / 2.0);
  size_t y = (size_t) round((resized.rows - kModelInputDim_) / 2.0);
  cv::Rect crop_region(x, y, kModelInputDim_, kModelInputDim_);
  cv::Mat center_cropped = resized(crop_region);

  // Normalization
  center_cropped.convertTo(normalized, CV_32FC3, 1/255.0);
  normalized -= cv::Scalar(0.485, 0.456, 0.406);
  cv::divide(normalized, cv::Scalar(0.229, 0.224, 0.225), normalized);

  if (kCondition_ == LoaderCondition::DecodeResizeNorm)
    return;

  std::vector<cv::Mat> channels(3);
  for (size_t i = 0, offset = 0; i < channels.size(); i++) {
    channels[i] = cv::Mat(normalized.rows, normalized.cols, CV_32FC1, output_buf + offset);
    offset += normalized.total();
  }
  cv::split(normalized, channels);
}
