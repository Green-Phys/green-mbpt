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

#include "green/embedding/seet_solver.h"

#include "green/impurity/impurity_solver.h"
#include "green/mbpt/orth.h"

namespace green::embedding {

  template <>
  mbpt::ztensor<4> seet_solver::compute_local_obj(const ztensor<5>& obj, const ztensor<4>& x_k) const {
    ztensor<4> obj_loc(_ft.sd().repn_fermi().nts(), _ns, _nso, _nso);
    for (size_t it = 0; it < obj.shape()[0]; ++it) {
      for (size_t is = 0; is < obj.shape()[1]; ++is) {
        auto obj_full = _bz_utils.ibz_to_full(obj(it, is));
        auto x_k_full = x_k.shape()[1] == _bz_utils.nk() ? x_k(is).copy() : _bz_utils.ibz_to_full(x_k(is));
        for (size_t ik = 0; ik < obj_full.shape()[0]; ++ik) {
          mbpt::matrix(obj_loc(it, is)) += mbpt::matrix(x_k_full(ik)) * mbpt::matrix(obj_full(ik)) * mbpt::matrix(x_k_full(ik)).adjoint();
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
      auto x_k_full = x_k.shape()[1] == _bz_utils.nk() ? x_k(is).copy() : _bz_utils.ibz_to_full(x_k(is));
      for (size_t ik = 0; ik < obj_full.shape()[0]; ++ik) {
        mbpt::matrix(obj_loc(is)) += mbpt::matrix(x_k_full(ik)) * mbpt::matrix(obj_full(ik)) * mbpt::matrix(x_k_full(ik)).adjoint();
      }
    }
    obj_loc /= _bz_utils.nk();
    return obj_loc;
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
      sigma_loc_new << sigma_loc_new_;
    }
    MPI_Bcast(sigma_inf_loc_new.data(), sigma_inf_loc_new.size(), MPI_CXX_DOUBLE_COMPLEX, 0, utils::context.global);
    MPI_Bcast(sigma_loc_new.data(), sigma_loc_new.size(), MPI_CXX_DOUBLE_COMPLEX, 0, utils::context.global);

    for (size_t is = 0; is < g.object().shape()[1]; ++is) {
      auto x_k = _x_inv_k.shape()[1] == _bz_utils.nk() ? _bz_utils.full_to_ibz(_x_inv_k(is)) : _x_inv_k(is).copy();
      for (size_t ik = 0; ik < g.object().shape()[2]; ++ik) {
        auto x_k_m = mbpt::matrix(x_k(ik));
        mbpt::matrix(sigma_inf(is, ik)) += x_k_m * mbpt::matrix(sigma_inf_loc_new(is)) * x_k_m.adjoint();
        sigma_tau.fence();
        for (size_t it = utils::context.node_rank; it < g.object().shape()[0]; it += utils::context.node_size) {
          mbpt::matrix(sigma_tau.object()(it, is, ik)) += x_k_m * mbpt::matrix(sigma_loc_new(it, is)) * x_k_m.adjoint();
        }
        sigma_tau.fence();
      }
    }
  }

  void seet_inner_solver::solve(G_type& g, S1_type& sigma_inf, St_type& sigma_tau) {
    h5pp::archive ar(_weak_results_file, "r");
    ar[_base_path + "/Sigma1"] >> sigma_inf;
    sigma_tau.fence();
    if (!utils::context.node_rank) {
      ar[_base_path + "/Selfenergy/data"] >> sigma_tau.object();
    }
    sigma_tau.fence();
    ar.close();
  }
}
