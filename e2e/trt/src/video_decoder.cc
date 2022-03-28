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
#include "libavfilter/avfilter.h"
#include "libavfilter/buffersrc.h"
#include "libavfilter/buffersink.h"
#include "libavcodec/avcodec.h"
}

#include "video_decoder.h"

static int open_codec_context(int *stream_idx, AVCodecContext **dec_ctx,
    AVFormatContext *fmt_ctx, enum AVMediaType type) {
  int ret, stream_index;
  AVStream *st;
  const AVCodec *dec = NULL;
  AVDictionary *opts = NULL;

  ret = av_find_best_stream(fmt_ctx, type, -1, -1, NULL, 0);
  if (ret < 0)
    throw std::runtime_error("Couldn't find stream in input file");
  stream_index = ret;
  st = fmt_ctx->streams[stream_index];

  /* find decoder for the stream */
  dec = avcodec_find_decoder(st->codecpar->codec_id);
  if (!dec)
    throw std::runtime_error("Coudln't find codec");

  /* Allocate a codec context for the decoder */
  *dec_ctx = avcodec_alloc_context3(dec);
  if (!*dec_ctx)
    throw std::runtime_error("Couldn't allocate codec context");

  /* Copy codec parameters from input stream to output codec context */
  if ((ret = avcodec_parameters_to_context(*dec_ctx, st->codecpar)) < 0)
    throw std::runtime_error("Failed to copy codec paramaeters to decoder context");

  /* Init the decoders, with or without reference counting */
  int refcount = 0; // TODO: figure out wtf this is
  av_dict_set(&opts, "refcounted_frames", refcount ? "1" : "0", 0);
  if ((ret = avcodec_open2(*dec_ctx, dec, &opts)) < 0)
    throw std::runtime_error("Failed to open codec");
  *stream_idx = stream_index;

  return 0;
}


VideoDecoder::VideoDecoder(
    const std::string& kFname, const enum PixelFormat dst_pfmt,
    const size_t kOutputResol, const size_t kNbFrames,
    CropRegion region, LoaderCondition cond,
    const bool kDoResize) :
    dst_pix_fmt_(PixFormat::GetLibavPixelFormat(dst_pfmt)),
    kPlanar_(PixFormat::IsPlanar(dst_pfmt)),
    kOutWidth_(kOutputResol), kOutHeight_(kOutputResol),
    kNbFrames_(kNbFrames),
    verbose_(false), kCondition_(cond) {
  // open input file, and allocate format context
  if (avformat_open_input(&fmt_ctx_, kFname.c_str(), NULL, NULL) < 0)
    throw std::invalid_argument("Could't open file");
  // retrieve stream information
  if (avformat_find_stream_info(fmt_ctx_, NULL) < 0)
    throw std::invalid_argument("Couldn't find stream info");
  if (open_codec_context(&video_stream_idx_, &video_dec_ctx_, fmt_ctx_, AVMEDIA_TYPE_VIDEO) < 0)
    throw std::invalid_argument("Couldn't find a video stream");

  // kNbFrames_ = fmt_ctx_->streams[video_stream_idx_]->nb_frames;
  kInWidth_ = video_dec_ctx_->width;
  kInHeight_ = video_dec_ctx_->height;
  in_pix_fmt_ = video_dec_ctx_->pix_fmt;

  if (verbose_)
    av_dump_format(fmt_ctx_, 0, kFname.c_str(), 0);

  frame_ = av_frame_alloc();
  if (!frame_)
    throw std::runtime_error("Couldn't allocate frame");

  pkt_ = av_packet_alloc();

  // FIXME
  // Set up swscale
  if (kDoResize) {
    sws_ctx_ = sws_getContext(region.right - region.left,
                              region.bottom - region.top,
                              in_pix_fmt_,
                              kOutWidth_, kOutHeight_, dst_pix_fmt_,
                              SWS_FAST_BILINEAR, NULL, NULL, NULL);
  } else {
    sws_ctx_ = NULL;
  }
  cropper_ = new Cropper(kInWidth_, kInHeight_, in_pix_fmt_, region);
}


VideoDecoder::~VideoDecoder() {
  delete cropper_;
  avformat_close_input(&fmt_ctx_);
  avformat_free_context(fmt_ctx_);
  fmt_ctx_ = NULL;

  avcodec_free_context(&video_dec_ctx_);
  av_free(video_dec_ctx_);

  av_freep(&frame_->data[0]);
  av_frame_free(&frame_);

  av_packet_free(&pkt_);
  sws_freeContext(sws_ctx_);
}


int VideoDecoder::DecodePacket(int *got_frame, int cached) {
  int ret = 0;
  int decoded = pkt_->size;

  *got_frame = 0;
  if (pkt_->stream_index == video_stream_idx_) {
    ret = avcodec_decode_video2(video_dec_ctx_, frame_, got_frame, pkt_);
    if (ret < 0)
      throw std::runtime_error("Error decoding frame");

    if (*got_frame) {
      if (frame_->width != kInWidth_ || frame_->height != kInHeight_ || frame_->format != in_pix_fmt_)
        throw std::runtime_error("Frame {width,height,pix_fmt} changed");
      video_frame_count_++;
    }
  }

  if (*got_frame && refcount)
    av_frame_unref(frame_);
  return decoded;
}


// FIXME: Optimized vs not?
void VideoDecoder::ProcessFrame(cv::Mat *converted) {
  if (kCondition_ == LoaderCondition::DecodeOnly)
    return;

  AVFrame *cropped = cropper_->crop(frame_);
  if (kCondition_ == LoaderCondition::DecodeCrop) {
    av_frame_unref(cropped);
    return;
  }

  if (sws_ctx_ != NULL) {
    uint8_t *dst_data[4] = {NULL};
    int dst_linesize[4] = {kOutWidth_, kOutWidth_ ,kOutWidth_, 0};
    const size_t kChannelSize = kOutWidth_ * kOutHeight_;
    if (kPlanar_) {
      for (size_t ch = 0; ch < 3; ch++) {
        size_t offset = ch * kChannelSize;
        dst_data[ch] = converted->data + offset;
      }
    } else {
      dst_data[0] = converted->data;
    }
    sws_scale(sws_ctx_,
              (const uint8_t * const*) cropped->data,
              cropped->linesize, 0, cropped->height,
              dst_data, dst_linesize);

    /*cv::Mat c1(kOutWidth_, kOutHeight_, CV_8UC1, converted->data);
    cv::Mat c2(kOutWidth_, kOutHeight_, CV_8UC1, converted->data + kChannelSize);
    cv::Mat c3(kOutWidth_, kOutHeight_, CV_8UC1, converted->data + kChannelSize * 2);
    cv::Mat channels[3] = {c1, c2, c3};
    cv::Mat output;
    cv::merge(channels, 3, output);
    cv::imwrite("/afs/cs.stanford.edu/u/ddkang/test.png", output);
    throw std::runtime_error("hi");*/
  } else {
    // TODO: totally fucked
    /*for (size_t ch = 0; ch < 3; ch++) {
      for (size_t row = 0; row < cropped->height; row++) {
        std::copy(cropped->data[ch], cropped->data[ch] + cropped->width,
                  dst_data_[ch]);
      }
    }*/
  }
  av_frame_unref(cropped);
  if (kCondition_ == LoaderCondition::DecodeResize)
    return;
}


void VideoDecoder::DecodeAll(uint8_t *output) {
  std::vector<cv::Mat> return_frames(kNbFrames_);
  int got_frame;
  int sizes[3];
  if (kPlanar_) {
    sizes[0] = 3; sizes[1] = kOutHeight_; sizes[2] = kOutWidth_;
  } else {
    sizes[0] = kOutHeight_; sizes[1] = kOutWidth_; sizes[2] = 3;
  }
  const size_t kFrameSize = 3 * kOutWidth_ * kOutHeight_;
  for (size_t i = 0; i < return_frames.size(); i++) {
    uint8_t *data = output + kFrameSize * i;
    return_frames[i] = cv::Mat(3, sizes, CV_8UC1, data);
  }

  while (av_read_frame(fmt_ctx_, pkt_) >= 0) {
    AVPacket *orig_pkt = av_packet_clone(pkt_);
    do {
      const int ret = DecodePacket(&got_frame, 0);
      if (got_frame) {
        // video_frame_count is incremented before this
        if (video_frame_count_ - 1 < return_frames.size())
          ProcessFrame(&return_frames[video_frame_count_ - 1]);
      }
      if (ret < 0)
        break;
      pkt_->data += ret;
      pkt_->size -= ret;
    } while (pkt_->size > 0);
    av_packet_unref(orig_pkt);
  }

  pkt_->data = NULL;
  pkt_->size = 0;
  do {
    DecodePacket(&got_frame, 1);
    if (got_frame) {
      if (video_frame_count_ - 1 < return_frames.size())
        ProcessFrame(&return_frames[video_frame_count_ - 1]);
    }
  } while (got_frame);
}
