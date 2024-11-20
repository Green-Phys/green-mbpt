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

#ifndef MBPT_SEET_SOLVER_H
#define MBPT_SEET_SOLVER_H

#include <green/grids/transformer_t.h>
#include <green/impurity/impurity_solver.h>
#include <green/ndarray/ndarray.h>
#include <green/ndarray/ndarray_math.h>
#include <green/params/params.h>
#include <green/symmetry/symmetry.h>
#include <green/utils/mpi_shared.h>
#include <green/utils/mpi_utils.h>
#include <green/utils/timing.h>
#include <mpi.h>

#include "common_defs.h"
#include "df_integral_t.h"
#include "kernel_factory.h"
#include "kernels.h"
#include "mbpt_q0_utils_t.h"

namespace green::mbpt {

  class impurity_params {
  private:
    size_t _orb_start;
  };

  /**
   * @brief Self-energy embedding solver class. Computes contribution into a self-energy
   * from
   */
  class seet_solver {
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;
    using G_type     = utils::shared_object<ztensor<5>>;
    using S1_type    = ztensor<4>;
    using St_type    = utils::shared_object<ztensor<5>>;

  public:
    /**
     * Class constructor
     *
     * @param p           -- simulation parameters
     * @param ft          -- imaginary time transformer
     * @param Gk          -- Green's function in (tau, kcell, nao, nao) domain
     * @param Sigma       -- Self-energy in (tau, kcell, nao, nao) domain
     * @param bz_utils    -- Brillouin zone utilities
     * @param second_only -- Whether do GW or only second-order direct diagram
     */
    seet_solver(const params::params& p, const grids::transformer_t& ft, const bz_utils_t& bz_utils, const ztensor<4>& H_k,
                const ztensor<4>& S_k, const double& mu) :
        _ft(ft), _bz_utils(bz_utils), _ovlp_k(S_k), _h_core_k(H_k), _mu(mu),
        _solver(p, ft, bz_utils) {
      h5pp::archive ar(p["input_file"]);
      ar["params/nao"] >> _nao;
      ar["params/nso"] >> _nso;
      ar["params/ns"] >> _ns;
      ar.close();
      ar.open(p["seet_input"]);
      ztensor<3> x_;
      ar["nimp"] >> _nimp;
      ar["X_k"] >> x_;
      _x_k.resize(_ns + x_.shape());
      for (size_t is = 0; is < _ns; ++is) _x_k(is) << x_;
      ar["X_inv_k"] >> x_;
      _x_inv_k.resize(_ns + x_.shape());
      for (size_t is = 0; is < _ns; ++is) _x_inv_k(is) << x_;
      ar.close();
    }

    /**
     * Solve SEET equations for Self-energy
     */
    void solve(G_type& g, S1_type&, St_type& sigma_tau);

  private:
    /**
     *
     * @param g object on time and k grid
     * @param x_k transformation matrices to orthogonal basis
     * @return local object in orthogonal basis
     */
    template <size_t N>
    ztensor<N - 1> compute_local_obj(const ztensor<N>& g, const ztensor<4>& x_k) const {
      throw std::runtime_error("Only implemented for N=4 and N=5");
    }

    std::tuple<ztensor<3>, ztensor<3>, ztensor<3>, ztensor<4>, ztensor<4>> project_to_as(const ztensor<3>& S, const ztensor<3>& H,
                                                                                         const ztensor<3>& S_1,
                                                                                         const ztensor<4>& G,
                                                                                         const ztensor<4>& Sigma,
                                                                                         const ztensor<2>& UU) const;

    const grids::transformer_t&        _ft;
    const bz_utils_t&                  _bz_utils;
    const ztensor<4>&                  _ovlp_k;
    const ztensor<4>&                  _h_core_k;
    const double&                      _mu;
    ztensor<4>                         _x_k;
    ztensor<4>                         _x_inv_k;
    size_t                             _nao;
    size_t                             _nso;
    size_t                             _ns;
    size_t                             _nimp;
    impurity::impurity_solver          _solver;
  };

}  // namespace green::mbpt

#endif  // MBPT_SEET_SOLVER_H
