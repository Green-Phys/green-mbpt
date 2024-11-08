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

#include "green/mbpt/seet_solver.h"

#include "green/impurity/impurity_solver.h"
#include "green/mbpt/orth.h"

namespace green::mbpt {

  template <>
  ztensor<4> seet_solver::compute_local_obj(const ztensor<5>& obj, const ztensor<4>& x_k) const {
    ztensor<4> obj_loc(_ft.sd().repn_fermi().nts(), _ns, _nso, _nso);
    for (size_t it = 0; it < obj.shape()[0]; ++it) {
      for (size_t is = 0; is < obj.shape()[1]; ++is) {
        auto obj_full = _bz_utils.ibz_to_full(obj(it, is));
        auto x_k_full = _bz_utils.ibz_to_full(x_k(is));
        for (size_t ik = 0; ik < obj_full.shape()[0]; ++ik) {
          matrix(obj_loc(it, is)) += matrix(x_k_full(ik)) * matrix(obj_full(ik)) * matrix(x_k_full(ik)).adjoint();
        }
      }
    }
    obj_loc /= _bz_utils.nk();
    return obj_loc;
  }
  template <>
  ztensor<3> seet_solver::compute_local_obj(const ztensor<4>& obj, const ztensor<4>& x_k) const {
    ztensor<3> obj_loc(_ns, _nso, _nso);
    for (size_t is = 0; is < obj.shape()[0]; ++is) {
      auto obj_full = _bz_utils.ibz_to_full(obj(is));
      auto x_k_full = _bz_utils.ibz_to_full(x_k(is));
      for (size_t ik = 0; ik < obj_full.shape()[0]; ++ik) {
        matrix(obj_loc(is)) += matrix(x_k_full(ik)) * matrix(obj_full(ik)) * matrix(x_k_full(ik)).adjoint();
      }
    }
    obj_loc /= _bz_utils.nk();
    return obj_loc;
  }

  std::tuple<ztensor<3>, ztensor<4>> seet_solver::solve_impurity(size_t imp_n, const ztensor<3>& ovlp, const ztensor<3>& h_core,
                                                                 const ztensor<3>& sigma_inf, const ztensor<4>& sigma,
                                                                 const ztensor<4>& g_w) const {
    // return std::make_tuple(sigma_inf_new, sigma_new);
    return std::make_tuple(ztensor<3>(), ztensor<4>());
  }

  void seet_solver::solve(G_type& g, S1_type& sigma_inf, St_type& sigma_tau) {
    ztensor<4> g_loc         = compute_local_obj(g.object(), _x_inv_k);
    ztensor<4> sigma_loc     = compute_local_obj(sigma_tau.object(), _x_k);
    ztensor<3> sigma_inf_loc = compute_local_obj(sigma_inf, _x_k);
    ztensor<3> ovlp_loc      = compute_local_obj(_ovlp_k, _x_k);
    ztensor<3> h_core_loc    = compute_local_obj(_h_core_k, _x_k);
    ztensor<3> sigma_inf_loc_new(sigma_inf_loc.shape());
    ztensor<4> sigma_loc_new(sigma_loc.shape());
    // loop over all impurities
    if (!utils::context.global_rank) {
      auto [sigma_inf_loc_new_, sigma_loc_new_] = _solver.solve(_mu, ovlp_loc, h_core_loc, sigma_inf_loc, sigma_loc, g_loc);
      sigma_inf_loc_new << sigma_inf_loc_new_;
      sigma_loc_new << sigma_loc;
    }
    MPI_Bcast(sigma_inf_loc_new.data(), sigma_inf_loc_new.size(), MPI_CXX_DOUBLE_COMPLEX, 0, utils::context.global);
    MPI_Bcast(sigma_loc_new.data(), sigma_loc_new.size(), MPI_CXX_DOUBLE_COMPLEX, 0, utils::context.global);
    for (size_t is = 0; is < g.object().shape()[1]; ++is) {
      for (size_t ik = 0; ik < g.object().shape()[2]; ++ik) {
        auto x_k = matrix(_x_inv_k(is, ik));
        matrix(sigma_inf(is, ik)) += x_k * matrix(sigma_inf_loc_new(is)) * x_k.adjoint();
        sigma_tau.fence();
        for (size_t it = utils::context.internode_rank; it < g.object().shape()[2]; it += utils::context.internode_size) {
          matrix(sigma_tau.object()(it, is, ik)) += x_k * matrix(sigma_loc_new(it, is)) * x_k.adjoint();
        }
        sigma_tau.fence();
      }
    }
  }
}