#ifndef CRITERION_H_
#define CRITERION_H_

#include <algorithm>
#include <cmath>
#include <iterator>
#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>


// Adapted from
// https://codereview.stackexchange.com/questions/177973/softmax-function-implementation
// Note that this implicitly assumes dest and beg have the same type, can be changed
template <typename It>
static void softmax(It beg, It end, It dest) {
  using VType = typename std::iterator_traits<It>::value_type;

  /*VType max_el = -10000, exp_tot = 0.;
  for (auto beg_cpy = beg, dest_cpy = dest; beg_cpy < end; beg_cpy++, dest_cpy++) {
    max_el = std::max(max_el, *beg_cpy);
  }

  for (auto beg_cpy = beg, dest_cpy = dest; beg_cpy < end; beg_cpy++, dest_cpy++) {
    *dest_cpy = std::exp(*beg_cpy - max_el);
    exp_tot += *dest_cpy;
  }

  for (auto beg_cpy = beg, dest_cpy = dest; beg_cpy < end; beg_cpy++, dest_cpy++) {
    *dest_cpy /= exp_tot;
  }*/

  auto dest_start = dest;
  auto beg_cpy = beg, end_cpy = end;
  auto const kMaxEl { *std::max_element(beg_cpy, end_cpy) };

  beg_cpy = beg; end_cpy = end;
  VType exptot = 0;
  std::transform(
      beg_cpy, end_cpy, dest,
      [&](VType x){
        auto ex = std::exp(x - kMaxEl);
        exptot += ex;
        return ex;
      });

  std::transform(
      dest_start, dest_start + (end - beg), dest_start,
      std::bind2nd(std::divides<VType>(), exptot));
}

class Criterion {
 public:
  virtual std::vector<bool> filter(const size_t kNbEl, std::vector<float> data) const = 0;
};

class MaxCriterion : public Criterion {
 private:
  const float kCutoff_;

 public:
  MaxCriterion(const float kCutoff) : kCutoff_(kCutoff) {} ;

  std::vector<bool> filter(const size_t kNbEl, std::vector<float> data) const;
};

#endif // CRITERION_H_
