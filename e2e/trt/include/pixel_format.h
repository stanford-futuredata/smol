#ifndef PIXEL_FORMAT_
#define PIXEL_FORMAT_

extern "C" {
#include "libavutil/samplefmt.h"
}

enum class PixelFormat {
 PLANAR_RGB, PLANAR_BGR, PLANAR_YUV,
 PACKED_RGB, PACKED_BGR, PACKED_YUV
};

namespace PixFormat {
// C++ class enums don't allow class methods
static AVPixelFormat GetLibavPixelFormat(PixelFormat pfmt) {
  switch(pfmt) {
    // These need processing downstream
    // FIXME: RGB/BGR doesn't do what I think it does lol
    case PixelFormat::PLANAR_RGB:
      return AV_PIX_FMT_GBRP;
    case PixelFormat::PLANAR_BGR:
      return AV_PIX_FMT_GBRP;
    case PixelFormat::PLANAR_YUV:
      return AV_PIX_FMT_YUV444P;

    case PixelFormat::PACKED_RGB:
      return AV_PIX_FMT_RGB24;
    case PixelFormat::PACKED_BGR:
      return AV_PIX_FMT_BGR24;

    // FFmpeg doesn't support packed 4:4:4 YUV?
    default:
      throw std::runtime_error("Not implemented");
  }
}

static bool IsPlanar(PixelFormat pfmt) {
  switch(pfmt) {
    case PixelFormat::PLANAR_RGB:
    case PixelFormat::PLANAR_BGR:
    case PixelFormat::PLANAR_YUV:
      return true;

    case PixelFormat::PACKED_RGB:
    case PixelFormat::PACKED_BGR:
      return false;

    default:
      throw std::runtime_error("Not implemented");
  }
}

} // namespace PixFormat

#endif // PIXEL_FORMAT_
