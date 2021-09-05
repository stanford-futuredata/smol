#ifndef CROPPER_H_
#define CROPPER_H_

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
}


class CropRegion {
 public:
  int left, top, right, bottom;
  CropRegion(int left, int top, int right, int bottom) :
      left(left), top(top), right(right), bottom(bottom) {}
};


class Cropper {
 private:
  AVFilterContext *buffersink_ctx;
  AVFilterContext *buffersrc_ctx;
  AVFilterGraph *filter_graph;
  AVFrame *f;
  AVFilterInOut *inputs, *outputs;
  char args[512];

 public:
  Cropper(const int kInWidth_, const int kInHeight_, AVPixelFormat fmt, CropRegion region) {
    filter_graph = avfilter_graph_alloc();
    f = av_frame_alloc();
    int ret;

    snprintf(args, sizeof(args),
             "buffer=video_size=%dx%d:pix_fmt=%d:time_base=1/1:pixel_aspect=0/1[in];"
             "[in]crop=x=%d:y=%d:out_w=%d:out_h=%d[out];"
             "[out]buffersink",
             kInWidth_, kInHeight_, fmt,
             region.left, region.top,
             region.right - region.left, region.bottom - region.top);

    ret = avfilter_graph_parse2(filter_graph, args, &inputs, &outputs);
    if (ret < 0) throw std::runtime_error("Failed to parse filter graph");
    assert(inputs == NULL && outputs == NULL);
    ret = avfilter_graph_config(filter_graph, NULL);
    if (ret < 0) throw std::runtime_error("Config graph failed");

    buffersrc_ctx = avfilter_graph_get_filter(filter_graph, "Parsed_buffer_0");
    buffersink_ctx = avfilter_graph_get_filter(filter_graph, "Parsed_buffersink_2");
    assert(buffersrc_ctx != NULL);
    assert(buffersink_ctx != NULL);
  }

  AVFrame* crop(const AVFrame *in) {
    int ret;
    av_frame_ref(f, in);
    ret = av_buffersrc_add_frame(buffersrc_ctx, f);
    if (ret < 0) throw std::runtime_error("Buffer src failed");
    ret = av_buffersink_get_frame(buffersink_ctx, f);
    if (ret < 0) throw std::runtime_error("Buffer sink failed");

    return f;
  }

  ~Cropper() {
    avfilter_graph_free(&filter_graph);
    av_frame_free(&f);
  }
};

#endif // CROPPER_H_
