/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef GREEN_MBPT_TENSOR_TEST_H
#define GREEN_MBPT_TENSOR_TEST_H

#include <green/ndarray/ndarray.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <cmath>

// Catch2 Matcher class that checks proximity of two tensors
template <typename T, size_t Dim>
class IsCloseToMatcher : public Catch::Matchers::MatcherBase<green::ndarray::ndarray<T, Dim>> {
  using tensor_t = green::ndarray::ndarray<T, Dim>;
  tensor_t const& ref;
  double          tol;

public:
  template <typename R>
  IsCloseToMatcher(R const& ref, double tol) : ref(ref), tol(tol) {}

  bool match(tensor_t const& x) const override {
    if (x.shape() != ref.shape()) return false;
    bool res =
        std::equal(x.data(), x.data() + x.size(), ref.data(), [this](T const& a, T const& b) { return std::abs(a - b) < tol; });
    return res;
  }

  virtual std::string describe() const override {
    std::ostringstream ss;
    ss << "is close to Tensor with leading dimension [" << ref.shape()[0] << "] (tol = " << tol << ")";
    return ss.str();
  }
};

template <typename T, size_t Dim>
inline IsCloseToMatcher<T, Dim> IsCloseTo(green::ndarray::ndarray<T, Dim> const& ref, double tol = 1e-10) {
  return IsCloseToMatcher<T, Dim>(ref, tol);
}

#endif  // GREEN_MBPT_TENSOR_TEST_H
