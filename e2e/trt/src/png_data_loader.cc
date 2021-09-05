#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "spng.h"

#include "data_loader.h"

cv::Mat PNGDataLoader::DecodeImage(CompressedImage compressed) const {
  spng_ctx *ctx;
  struct spng_ihdr ihdr;
  size_t size;

  ctx = spng_ctx_new(0);
  if (ctx == NULL
      || spng_set_png_buffer(ctx, compressed.first, compressed.second)
      || spng_get_ihdr(ctx, &ihdr)
      || spng_decoded_image_size(ctx, SPNG_FMT_RGBA8, &size)) {
    spng_ctx_free(ctx);
    throw std::invalid_argument("Could not decode PNG");
  }

  //             rows         cols
  cv::Mat decoded(ihdr.height, ihdr.width, CV_8UC4), output;

  spng_decode_image(ctx, decoded.data, size, SPNG_FMT_RGBA8, SPNG_DECODE_USE_TRNS);
  cv::cvtColor(decoded, output, cv::COLOR_BGRA2BGR);
  return output;
}

// FIXME: same as naive
void PNGDataLoader::PreprocessImage(const cv::Mat& kRawImage, float *output_buf) const {
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





cv::Mat OptPNGDataLoader::DecodeImage(CompressedImage compressed) const {
  struct spng_ihdr ihdr;
  size_t size;

  spng_ctx *ctx = spng_ctx_new(0);
  if (ctx == NULL
      || spng_set_png_buffer(ctx, compressed.first, compressed.second)
      || spng_get_ihdr(ctx, &ihdr)
      || spng_decoded_image_size(ctx, SPNG_FMT_RGBA8, &size)) {
    spng_ctx_free(ctx);
    throw std::invalid_argument("Could not decode PNG");
  }

  //              rows         cols
  cv::Mat decoded(ihdr.height, ihdr.width, CV_8UC4), output;

  spng_decode_image(ctx, decoded.data, size, SPNG_FMT_RGBA8, SPNG_DECODE_USE_TRNS);
  cv::cvtColor(decoded, output, cv::COLOR_BGRA2BGR);
  return output;
}

void OptPNGDataLoader::PreprocessImage(const cv::Mat& kRawImage, float *output_buf) const {
  if (kCondition_ == LoaderCondition::DecodeOnly)
    return;
  cv::Mat resized;

  auto new_resol = RatioPreservingResize(kResizeDim_, kRawImage.cols, kRawImage.rows);
  cv::resize(kRawImage, resized, cv::Size(new_resol.first, new_resol.second));
  if (kCondition_ == LoaderCondition::DecodeResize)
    return;

  size_t x = (size_t) round((resized.cols - kModelInputDim_) / 2.0);
  size_t y = (size_t) round((resized.rows - kModelInputDim_) / 2.0);
  cv::Rect crop_region(x, y, kModelInputDim_, kModelInputDim_);
  cv::Mat center_cropped = resized(crop_region);

  std::vector<uint8_t> scratch(kModelInputDim_ * kModelInputDim_ * 3);
  std::vector<cv::Mat> channels(3);
  for (size_t i = 0, offset = 0; i < channels.size(); i++) {
    channels[i] = cv::Mat(center_cropped.rows, center_cropped.cols, CV_8UC1, scratch.data() + offset);
    offset += center_cropped.total();
  }
  cv::split(center_cropped, channels);

  for (size_t i = 0, offset = 0; i < channels.size(); i++) {
    for (size_t j = 0; j < kModelInputDim_ * kModelInputDim_; j++) {
      output_buf[offset] = map_[i][scratch[offset]];
      offset++;
    }
  }
}


void OptResizePNGDataLoader::PreprocessImage(const cv::Mat& kRawImage, float *output_buf) const {
  if (kCondition_ == LoaderCondition::DecodeOnly)
    return;

  cv::Mat resized, center_cropped;
  const size_t short_side = std::min(kRawImage.cols, kRawImage.rows);
  const size_t crop_size = (size_t) (short_side * kModelInputDim_ / (float) kResizeDim_);
  const auto new_resol = RatioPreservingResize(crop_size, kRawImage.cols, kRawImage.rows);
  const size_t x = (size_t) round((kRawImage.cols - new_resol.first) / 2.0);
  const size_t y = (size_t) round((kRawImage.rows - new_resol.second) / 2.0);
  cv::Rect crop_region(x, y, crop_size, crop_size);
  cv::resize(kRawImage(crop_region), center_cropped, cv::Size(kModelInputDim_, kModelInputDim_));
  if (kCondition_ == LoaderCondition::DecodeResize)
    return;

  std::vector<uint8_t> scratch(kModelInputDim_ * kModelInputDim_ * 3);
  std::vector<cv::Mat> channels(3);
  for (size_t i = 0, offset = 0; i < channels.size(); i++) {
    channels[i] = cv::Mat(center_cropped.rows, center_cropped.cols, CV_8UC1, scratch.data() + offset);
    offset += center_cropped.total();
  }
  cv::split(center_cropped, channels);

  for (size_t i = 0, offset = 0; i < channels.size(); i++) {
    for (size_t j = 0; j < kModelInputDim_ * kModelInputDim_; j++) {
      output_buf[offset] = map_[i][scratch[offset]];
      offset++;
    }
  }
}
