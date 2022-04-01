#ifndef VIDEO_DECODER_H_
#define VIDEO_DECODER_H_

#include <memory>
#include <stdint.h>
#include <string>
#include <vector>

#define __STDC_CONSTANT_MACROS 
extern "C" {
#include "libavutil/imgutils.h"
#include "libavutil/samplefmt.h"
#include "libavutil/timestamp.h"
#include "libavformat/avformat.h"
}


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "common.h"
#include "pixel_format.h"
#include "cropper.h"

class VideoDecoder {
 private:
  const bool verbose_;
  AVFormatContext *fmt_ctx_ = NULL;
  AVCodecContext *video_dec_ctx_ = NULL;
  AVFrame *frame_ = NULL;
  AVPacket *pkt_;

  LoaderCondition kCondition_;
  enum AVPixelFormat in_pix_fmt_;
  int video_stream_idx_ = -1;
  int video_frame_count_ = 0;
  // The following should be const
  size_t kNbFrames_;
  int kInWidth_, kInHeight_;


  // Output data
  Cropper *cropper_;

  const enum AVPixelFormat dst_pix_fmt_;
  const bool kPlanar_;
  const int kOutWidth_, kOutHeight_;
  struct SwsContext *sws_ctx_ = NULL;

 public:
  // IMPORTANT: The caller is responsible for all the files to have the same
  // number of frames
  VideoDecoder(
      const std::string& kFname, const enum PixelFormat dst_pfmt,
      const size_t kOutputResol, const size_t kNbFrames,
      CropRegion region, LoaderCondition cond,
      const bool kDoResize = true);
  ~VideoDecoder();
  void InitLookup();

  void ProcessFrame(cv::Mat *converted);
  void DecodePacket(AVPacket *pkt, std::vector<cv::Mat> &return_frames);
  void DecodeAll(uint8_t *output);
};

#endif // VIDEO_DECODER_H_
