/*
 * Copyright (c) 2023 University of Michigan
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

#ifndef MBPT_ORTH_H
#define MBPT_ORTH_H

#include <green/mbpt/except.h>

#include "common_defs.h"

using namespace std::string_literals;

namespace green::mbpt {

  inline void compute_S_sqrt(const ztensor<3>& Sk, ztensor<3>&& Sk_12_inv) {
    size_t ink     = Sk.shape()[0];
    size_t nso     = Sk.shape()[1];
    using Matrixcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    Eigen::SelfAdjointEigenSolver<Matrixcd> solver(nso);
    Eigen::FullPivLU<MatrixXcd>             lusolver(nso, nso);
    for (size_t ik = 0; ik < ink; ++ik) {
      Matrixcd S = matrix(Sk(ik));
      solver.compute(S);
      matrix(Sk_12_inv(ik)) =
          solver.eigenvectors() * (solver.eigenvalues().cwiseSqrt().asDiagonal()) * solver.eigenvectors().adjoint();
      matrix(Sk_12_inv(ik)) = lusolver.compute(matrix(Sk_12_inv(ik))).inverse().eval();
    }
  }

  /**
   * Construct transform to orthogonal basis
   *
   * @param orth_type
   * @param Sk
   * @param Hk
   * @param Sigma_1_k
   * @param X_k
   */
  inline void orth(const std::string& orth_type, const ztensor<4>& Sk, const ztensor<4>& Hk, const ztensor<4>& Sigma_1_k,
                   ztensor<4>& X_k) {
    if (orth_type == "symm") {
      compute_S_sqrt(Sk(0), X_k(0));
      for (size_t is(1); is < X_k.shape()[0]; ++is) {
        X_k(is) << X_k(0);
      }
    } else {
      throw mbpt_orth_error("Unknown orthogonalization type");
    }
  }
}  // namespace green::mbpt

#endif  // MBPT_ORTH_H
