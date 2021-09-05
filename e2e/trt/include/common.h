#ifndef COMMON_H_
#define COMMON_H_

#include <utility>
#include <vector>

#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include "folly/MPMCQueue.h"

// Due to weird JPEG fuckery, the JPEG routines free the compressed image files (???)
// As a result, pass the raw pointers, not anything smarter
typedef std::pair<uint8_t *, size_t> CompressedImage;
// typedef std::unique_ptr<std::vector<float> > Batch;
typedef std::vector<float, thrust::system::cuda::experimental::pinned_allocator<float> > BatchBase;
typedef std::unique_ptr<BatchBase> Batch;
typedef std::tuple<
    Batch, size_t,
    float *, size_t, folly::MPMCQueue<Batch> *> QueueData;


class LoaderCondition {
 public:
  enum Value {
    DecodeOnly,
    DecodeCrop, // Video only
    DecodeResize, // Includes crop for video
    DecodeResizeNorm,
    All
  };

  LoaderCondition() = default;
  constexpr LoaderCondition(Value val) : val_(val) {}

  // Enables switch
  operator Value() const { return val_; }
  explicit operator bool() = delete;

  static const Value GetVal(std::string val) {
    if (val == "decode-only") {
      return DecodeOnly;
    } else if (val == "decode-crop") {
      return DecodeCrop;
    } else if (val == "decode-resize") {
      return DecodeResize;
    } else if (val == "decode-resize-norm") {
      return DecodeResizeNorm;
    } else if (val == "all") {
      return All;
    } else {
      throw std::invalid_argument("Wrong decoder condition");
    }
  }

 private:
  Value val_;
};



#endif // COMMON_H_
