#include <iostream>
#include <iterator>
#include <fstream>

#define __STDC_CONSTANT_MACROS 
extern "C" {
#include "libavutil/imgutils.h"
#include "libavutil/samplefmt.h"
#include "libavutil/timestamp.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
}

#include "video_data_loader.h"
#include "video_decoder.h"

// Is there a way to not copy this? doesn't matter that much
CompressedImage VideoDataLoader::LoadCompressedImageFromFile(const std::string& kFileName) const {
  std::ifstream file(kFileName, std::ios::binary | std::ios::in);
  file.unsetf(std::ios::skipws);

  // Get file size
  file.seekg(0, std::ios::end);
  std::streampos file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  uint8_t *ret = (uint8_t *) malloc(file_size);
  std::copy(std::istream_iterator<uint8_t>(file), std::istream_iterator<uint8_t>(), ret);
  file.close();
  return std::make_pair(ret, file_size);
}



// FIXME: pixel format, resol, nbframes
std::vector<cv::Mat> NaiveVidDataLoader::DecodeGOP(const std::string& kFileName) const {
  const size_t kNbFrames = 150;
  const size_t kFrameSize = 3 * kModelInputDim_ * kModelInputDim_;
  // FIXME: make this dumber
  uint8_t *output_buf = (uint8_t *) malloc(kFrameSize * kNbFrames);
  VideoDecoder decoder(
      kFileName,
      PixelFormat::PACKED_RGB, kModelInputDim_,
      kNbFrames,
      kRegion_,
      kCondition_);
  decoder.DecodeAll(output_buf);

  std::vector<cv::Mat> ret;
  for (size_t i = 0; i < kNbFrames; i++) {
    size_t offset = i * kFrameSize;
    ret.push_back(cv::Mat(kModelInputDim_, kModelInputDim_, CV_8UC3, output_buf + offset));
  }

  return ret;
}

void NaiveVidDataLoader::PreprocessGOP(const std::vector<cv::Mat>& kRawGOP, float *output_buf) const {
  cv::Mat normalized;
  const size_t kFrameSize = 3 * kModelInputDim_ * kModelInputDim_;
  for (size_t frame = 0; frame < kRawGOP.size(); frame++) {
    kRawGOP[frame].convertTo(normalized, CV_32FC3, 1/255.0);
    normalized -= cv::Scalar(0.485, 0.456, 0.406);
    cv::divide(normalized, cv::Scalar(0.229, 0.224, 0.225), normalized);

    if (kCondition_ == LoaderCondition::DecodeResizeNorm)
      continue;

    std::vector<cv::Mat> channels(3);
    for (size_t ch = 0, offset = frame * kFrameSize; ch < 3; ch++) {
      channels[ch] = cv::Mat(normalized.rows, normalized.cols, CV_32FC1, output_buf + offset);
      offset += normalized.total();
    }
    cv::split(normalized, channels);
  }
  // FIXME: due to the poor way the decoder was written, DecodeGOP allocates memory, which needs to
  // be freed
  free(kRawGOP[0].data);
}

void NaiveVidDataLoader::DecodeAndPreprocessGOP(const std::string& kFileName, float *output_buf) const {
  std::vector<cv::Mat> mats = DecodeGOP(kFileName);
  PreprocessGOP(mats, output_buf);

  /*const size_t kFrameSize = 3 * kModelInputDim_ * kModelInputDim_;
  std::ofstream fout("trt_frame.out", std::ios::out | std::ios::binary);
  fout.write((char *) output_buf, kFrameSize * sizeof(float));
  fout.close();
  throw std::runtime_error("hi");*/
}



std::vector<cv::Mat> OptimizedVidDataLoader::DecodeGOP(const std::string& kFileName) const {
}

void OptimizedVidDataLoader::PreprocessGOP(const std::vector<cv::Mat>& kRawGOP, float *output_buf) const {

}

// TODO: non-optimized loader w/o planar?
void OptimizedVidDataLoader::DecodeAndPreprocessGOP(const std::string& kFileName, float *output_buf) const {
  // FIXME: pixel format, resol, nbframes
  const size_t kNbFrames = 150;
  const size_t kChannelSize = kModelInputDim_ * kModelInputDim_;
  const size_t kFrameSize = 3 * kChannelSize;
  // FIXME: alias into output_buf?
  std::vector<uint8_t> tmp_buf(kNbFrames * kFrameSize);
  VideoDecoder decoder(
      kFileName,
      PixelFormat::PLANAR_RGB, kModelInputDim_,
      kNbFrames,
      kRegion_,
      kCondition_);
  decoder.DecodeAll(tmp_buf.data());

  if (kCondition_ == LoaderCondition::DecodeResize)
    return;

  for (size_t i = 0; i < kNbFrames; i++) {
    size_t out_offset = i * kFrameSize;
    for (size_t ch = 0; ch < 3; ch++) {
      const size_t in_chan = (ch + 2) % 3;
      size_t in_offset = i * kFrameSize + kChannelSize * in_chan;
      for (size_t j = 0; j < kChannelSize; j++) {
        output_buf[out_offset++] = map_[ch][tmp_buf[in_offset++]];
      }
    }
  }
}
