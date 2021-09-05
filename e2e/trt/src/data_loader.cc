#include <iostream>
#include <iterator>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "turbojpeg.h"
#include "jpeglib.h"

#include "data_loader.h"

CompressedImage DataLoader::LoadCompressedImageFromFile(const std::string& kFileName) const {
  std::ifstream file(kFileName, std::ios::binary | std::ios::in);
  file.unsetf(std::ios::skipws);

  // Get file size
  file.seekg(0, std::ios::end);
  std::streampos file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  uint8_t *ret = (uint8_t *) malloc(file_size);
  std::copy(std::istream_iterator<uint8_t>(file), std::istream_iterator<uint8_t>(), ret);
  return std::make_pair(ret, file_size);
}


cv::Mat OptimizedDataLoader::DecodeImage(CompressedImage compressed) const {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, compressed.first, compressed.second);
  (void) jpeg_read_header(&cinfo, TRUE);
  (void) jpeg_start_decompress(&cinfo);

  const size_t kOrigWidth = cinfo.output_width;
  const size_t kOrigHeight = cinfo.output_height;

  auto new_resol = RatioPreservingResize(kResizeDim_, cinfo.output_width, cinfo.output_height);
  size_t left = round((new_resol.first - kModelInputDim_) / 2);
  size_t top = round((new_resol.second - kModelInputDim_) / 2);
  size_t right = left + kModelInputDim_;
  size_t bottom = top + kModelInputDim_;
  float upscaling_factor = (cinfo.output_width > cinfo.output_height ? cinfo.output_height : cinfo.output_width) / (float) kResizeDim_;

  unsigned int adjusted_left = round(left * upscaling_factor);
  unsigned int adjusted_right = round(right * upscaling_factor);
  unsigned int adjusted_top = round(top * upscaling_factor);
  unsigned int adjusted_bottom = round(bottom * upscaling_factor);

  unsigned int crop_x_offset = adjusted_right - adjusted_left;
  unsigned int prev_adj_left = adjusted_left;
  unsigned int prev_adj_right = adjusted_right;
  jpeg_crop_scanline(&cinfo, &adjusted_left, &crop_x_offset);
  unsigned int new_adj_left = adjusted_left;
  unsigned int new_adj_right = new_adj_left + crop_x_offset;

  while (!(new_adj_left <= prev_adj_left && new_adj_right >= prev_adj_right)) {
    new_adj_left = adjusted_left - 8;
    if (new_adj_left < 0) {
      new_adj_left = 0;
    }
    new_adj_right = adjusted_right + 8;
    if (new_adj_right >= kOrigWidth)
      new_adj_right = kOrigWidth - 1;
    crop_x_offset = new_adj_right - new_adj_left;
    jpeg_crop_scanline(&cinfo, &new_adj_left, &crop_x_offset);
    new_adj_right = new_adj_left + crop_x_offset;
  }

  int left_crop = prev_adj_left - new_adj_left;
  int right_crop = new_adj_right - prev_adj_right;
  unsigned int crop_y_offset = adjusted_bottom - adjusted_top;
  unsigned int row_stride = crop_x_offset * cinfo.output_components;

  cv::Mat decoded_img(crop_y_offset, crop_x_offset, CV_8UC3);
  uint8_t *img_buf = decoded_img.ptr();
  jpeg_skip_scanlines(&cinfo, adjusted_top);

  size_t newRowIdx = 0;
  while (cinfo.output_scanline < adjusted_bottom) {
    size_t nSimultaneousRows = cinfo.rec_outbuf_height;
    uint8_t *buffer_array[nSimultaneousRows];
    for (size_t i = 0; i < nSimultaneousRows; i++)
      buffer_array[i] = img_buf + (newRowIdx * row_stride);
    (void) jpeg_read_scanlines(&cinfo, buffer_array, nSimultaneousRows);
    newRowIdx += nSimultaneousRows;
  }
  jpeg_abort_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  cv::Rect crop_region(
      left_crop, 0,
      decoded_img.cols - right_crop - left_crop, decoded_img.rows);
  return decoded_img(crop_region);
}


void OptimizedDataLoader::PreprocessImage(const cv::Mat& kRawImage, float *output_buf) const {
  if (kCondition_ == LoaderCondition::DecodeOnly)
    return;

  cv::Mat resized;
  if (kDoResize_) {
    resized.create(kModelInputDim_, kModelInputDim_, CV_8UC3);
    cv::resize(kRawImage, resized, cv::Size(kModelInputDim_, kModelInputDim_));
  } else {
    resized = kRawImage;
  }
  if (kCondition_ == LoaderCondition::DecodeResize)
    return;

  std::vector<uint8_t> scratch(kModelInputDim_ * kModelInputDim_ * 3);
  std::vector<cv::Mat> channels(3);
  for (size_t i = 0, offset = 0; i < channels.size(); i++) {
    channels[i] = cv::Mat(resized.rows, resized.cols, CV_8UC1, scratch.data() + offset);
    offset += resized.total();
  }
  cv::split(resized, channels);

  for (size_t i = 0, offset = 0; i < channels.size(); i++) {
    for (size_t j = 0; j < kModelInputDim_ * kModelInputDim_; j++) {
      output_buf[offset] = map_[i][scratch[offset]];
      offset++;
    }
  }

  // Verified correct
  /*resized.convertTo(normalized, CV_32FC3, 1/255.0);
  normalized -= cv::Scalar(0.485, 0.456, 0.406);
  cv::divide(normalized, cv::Scalar(0.229, 0.224, 0.225), normalized);

  std::vector<cv::Mat> channels(3);
  for (size_t i = 0, offset = 0; i < channels.size(); i++) {
    channels[i] = cv::Mat(normalized.rows, normalized.cols, CV_32FC1, output_buf + offset);
    offset += normalized.total();
  }
  cv::split(normalized, channels);*/

  // Verified correct
  /*std::vector<cv::Mat> ch_correct(3);
  cv::split(normalized, ch_correct);
  std::vector<float> outp_correct;
  for (size_t i = 0; i < ch_correct.size(); i++) {
    cv::Mat flat = ch_correct[i].reshape(1, 1);
    std::copy(flat.begin<float>(), flat.end<float>(), std::back_inserter(outp_correct));
  }*/
}


void DataLoader::DecodeAndPreproc(CompressedImage kCompressedBuf, float *output_buf) const {
  cv::Mat decoded = DecodeImage(kCompressedBuf);
  PreprocessImage(decoded, output_buf);
}

void DataLoader::LoadAndPreproc(const std::string& kFileName, float *output_buf) const {
  auto compressed = LoadCompressedImageFromFile(kFileName);
  cv::Mat decoded = DecodeImage(compressed);
  PreprocessImage(decoded, output_buf);
}

/*std::vector<float> OptimizedDataLoader::LoadAndPreproc(const std::string& kFileName) const {
  std::vector<float> output;
  output.reserve(kModelInputDim_ * kModelInputDim_ * 3);

  auto compressed = LoadCompressedImageFromFile(kFileName);
  cv::Mat decoded = DecodeImage(compressed);
  PreprocessImage(decoded, output.data());
  return output;
}*/
