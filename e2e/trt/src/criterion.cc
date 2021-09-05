#include <cassert>
#include <iostream>

#include "criterion.h"

std::vector<bool> MaxCriterion::filter(const size_t kNbEl, std::vector<float> data) const {
  assert(data.size() % kNbEl == 0);
  const size_t kSingleSize = data.size() / kNbEl;
  std::vector<bool> ret(kNbEl);

  #pragma omp parallel for
  for (size_t i = 0; i < kNbEl; i++) {
    auto beg = data.begin() + i * kSingleSize;
    auto end = data.begin() + (i + 1) * kSingleSize;
    softmax(beg, end, beg);

    // Softmax modifies the iterators
    beg = data.begin() + i * kSingleSize;
    end = data.begin() + (i + 1) * kSingleSize;

    const float kMaxEl { *std::max_element(beg, end) };
    if (kMaxEl < kCutoff_) {
      ret[i] = false;
    } else {
      ret[i] = true;
    }
  }

  return ret;
}
