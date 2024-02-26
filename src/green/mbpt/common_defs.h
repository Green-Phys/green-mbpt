/*
 * Copyright (c) 2024 University of Michigan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the “Software”), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef MBPT_COMMON_DEFS_H
#define MBPT_COMMON_DEFS_H

#include <green/grids/itime_mesh_t.h>
#include <green/ndarray/ndarray.h>

#include <Eigen/Dense>

#ifdef GREEN_CUSTOM_KERNEL_HEADER
#include GREEN_CUSTOM_KERNEL_HEADER
#endif

namespace green::mbpt {
  // Matrix types
  template <typename prec>
  using MatrixX   = Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXcf = Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXd  = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  // Matrix-Map types
  template <typename prec>
  using MMatrixX   = Eigen::Map<Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MMatrixXcd = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MMatrixXcf = Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MMatrixXd  = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  // Const Matrix-Map types
  template <typename prec>
  using CMMatrixX   = Eigen::Map<const Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXcd = Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXcf = Eigen::Map<const Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXd  = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  // Column type
  using column      = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1, Eigen::ColMajor>;
  using Mcolumn     = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1, Eigen::ColMajor>>;
  using CMcolumn    = Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1, Eigen::ColMajor>>;
  // time grid type
  using tau_mesh    = grids::itime_mesh_t;
  // Tensor types
  template <typename prec, size_t Dim>
  using tensor = green::ndarray::ndarray<prec, Dim>;
  template <size_t Dim>
  using ztensor = green::ndarray::ndarray<std::complex<double>, Dim>;
  template <size_t Dim>
  using ztensor_view = green::ndarray::ndarray<std::complex<double>, Dim>;
  template <size_t Dim>
  using ztensor_base = green::ndarray::ndarray<std::complex<double>, Dim>;
  template <size_t Dim>
  using ctensor = green::ndarray::ndarray<std::complex<float>, Dim>;
  template <size_t Dim>
  using dtensor = green::ndarray::ndarray<double, Dim>;
  template <size_t Dim>
  using ltensor = green::ndarray::ndarray<long, Dim>;
  template <size_t Dim>
  using itensor = green::ndarray::ndarray<int, Dim>;

  template <typename prec, typename = std::enable_if_t<std::is_same_v<prec, std::remove_const_t<prec>>>>
  auto matrix(green::ndarray::ndarray<prec, 2>& array) {
    return MMatrixX<prec>(array.data(), array.shape()[0], array.shape()[1]);
  }

  template <typename prec, typename = std::enable_if_t<std::is_same_v<prec, std::remove_const_t<prec>>>>
  auto matrix(green::ndarray::ndarray<prec, 2>&& array) {
    return MMatrixX<prec>(array.data(), array.shape()[0], array.shape()[1]);
  }

  template <typename prec>
  auto matrix(const green::ndarray::ndarray<const prec, 2>& array) {
    return CMMatrixX<prec>(const_cast<prec*>(array.data()), array.shape()[0], array.shape()[1]);
  }

  template <typename prec>
  auto matrix(green::ndarray::ndarray<const prec, 2>&& array) {
    return CMMatrixX<prec>(const_cast<prec*>(array.data()), array.shape()[0], array.shape()[1]);
  }

  template <typename prec>
  auto matrix(const green::ndarray::ndarray<prec, 2>& array) {
    return CMMatrixX<prec>(array.data(), array.shape()[0], array.shape()[1]);
  }

  template <size_t N>
  void make_hermitian(ndarray::ndarray<std::complex<double>, N>& X) {
    // check that two innermost dimensions form a matrix
    assert(X.shape()[N - 1] == X.shape()[N - 2]);
    // Dimension of the rest of arrays
    size_t dim1 = std::accumulate(X.shape().begin(), X.shape().end() - 2, 1ul, std::multiplies<size_t>());
    size_t nao  = X.shape()[N - 1];
    for (size_t i = 0; i < dim1; ++i) {
      MMatrixXcd Xm(X.data() + i * nao * nao, nao, nao);
      Xm = 0.5 * (Xm + Xm.conjugate().transpose().eval());
    }
  }

  template <typename T, size_t D>
  std::array<size_t, D + 1> operator+(const std::array<size_t, D>& a, T b) {
    std::array<size_t, D + 1> result;
    std::copy(a.begin(), a.end(), result.begin());
    result[D] = size_t(b);
    return result;
  }

  template <typename T, size_t D>
  std::array<size_t, D + 1> operator+(T b, const std::array<size_t, D>& a) {
    std::array<size_t, D + 1> result;
    std::copy(a.begin(), a.end(), result.begin() + 1);
    result[0] = size_t(b);
    return result;
  }

  enum scf_type { HF, GF2, GW };

  enum sigma_q0_treatment_e { ignore_G0, ewald_int, extrapolate };

  enum job_type {SC, WINTER, THERMODYNAMICS};

  enum kernel_type {
    CPU
#ifdef CUSTOM_KERNEL
    , GREEN_CUSTOM_KERNEL_ENUM
#endif
  };

  inline void define_parameters(params::params& p) {
    p.define<std::string>("dfintegral_hf_file", "Path to Hartree-Fock integrals", "df_hf_int");
    p.define<std::string>("dfintegral_file", "Path to integrals for high orfer theories", "df_int");
    p.define<double>("tolerance,tol", "Double precision tolerance for chemical potential search.", 1e-9);
    p.define<double>("BETA,beta", "Inverse temperature.");
    p.define<scf_type>("scf_type", "Self-consistency level.");
    p.define<bool>("P_sp", "Compute polarization in single precision", false);
    p.define<bool>("Sigma_sp", "Compute self-energy in single precision", false);
    p.define<int>("verbose", "Print verbose output.", 0);
    p.define<std::string>("high_symmetry_output_file", "Name of the file to store Wannier interpolated Green's function.",
                          "output_hs.h5");
    p.define<sigma_q0_treatment_e>("q0_treatment", "GW q=0 divergence treatment", ignore_G0);
    p.define<std::vector<job_type>>("jobs", "Jobs to run.", std::vector{SC});
    p.define<kernel_type>("kernel", "Type of the computing kernel.", CPU);
#ifdef GREEN_CUSTOM_KERNEL_HEADER
    custom_kernel_parameters(p);
#endif
  }
}  // namespace green::mbpt
#endif  // MBPT_COMMON_DEFS_H
