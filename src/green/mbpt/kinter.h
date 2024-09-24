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

#ifndef MBPT_KINTER_H
#define MBPT_KINTER_H

#include "common_defs.h"
#include "orth.h"

namespace green::mbpt {
  template <typename T, size_t N>
  tensor<std::remove_const_t<T>, N> transform_to_hs(const tensor<T, N>& in, const ztensor<2>& tr) {
    size_t                dim_rest  = std::accumulate(in.shape().begin() + 1, in.shape().end(), 1ul, std::multiplies<size_t>());
    size_t                in_1      = in.shape()[0];
    size_t                out_1     = tr.shape()[0];
    std::array<size_t, N> out_shape = in.shape();
    out_shape[0]                    = out_1;
    ztensor<N>  out(out_shape);
    CMMatrixXcd in_m(in.data(), in_1, dim_rest);
    MMatrixXcd  out_m(out.data(), out_1, dim_rest);
    out_m = matrix(tr) * in_m;
    return out;
  }

  template <typename Dyson>
  void wannier_interpolation(const Dyson& dyson_solver, const ztensor<4>& sigma_1,
                             const utils::shared_object<ztensor<5>>& sigma_tau, const h5pp::archive& input,
                             const std::string& results_file) {
    using tau_hs_t = utils::shared_object<ztensor<4>>;
    auto [nel, mu] = dyson_solver.find_mu(sigma_1, sigma_tau);
    // Init arrays
    dtensor<2> kmesh_hs;
    dtensor<2> rmesh;
    input["high_symm_path/k_mesh"] >> kmesh_hs;
    input["high_symm_path/r_mesh"] >> rmesh;
    size_t     nts   = sigma_tau.object().shape()[0];
    size_t     ns    = sigma_tau.object().shape()[1];
    size_t     ink   = sigma_tau.object().shape()[2];
    size_t     nso   = sigma_tau.object().shape()[3];
    size_t     nw    = nts - 2;
    size_t     nk    = dyson_solver.bz_utils().nk();
    size_t     hs_nk = kmesh_hs.shape()[0];
    ztensor<4> Hk_hs(ns, hs_nk, nso, nso);
    ztensor<4> Sk_hs(ns, hs_nk, nso, nso);
    {
      ztensor<3> tmp;
      input["high_symm_path/Hk"] >> tmp;
      for(int is = 0; is < ns; ++is) Hk_hs(is) << tmp;
      input["high_symm_path/Sk"] >> tmp;
      for(int is = 0; is < ns; ++is) Sk_hs(is) << tmp;
    }
    ztensor<3> Sigma_w(ink, nso, nso);
    ztensor<2> G_w(nso, nso);
    ztensor<2> G_w_hs(nso, nso);
    ztensor<2> transform(hs_nk, nk);
    // exponential from r to k_hs
    ztensor<2> exp_kr(hs_nk, rmesh.shape()[0]);
    // exponential from k to r
    ztensor<2> exp_rk(rmesh.shape()[0], nk);
    // (nts, ns, nk_hs, nao)
    tau_hs_t   g_tau_hs(nts, ns, kmesh_hs.shape()[0], nso);
    tau_hs_t   g_omega_hs(nw, ns, kmesh_hs.shape()[0], nso);
    // Compute transformation matricies
    for (size_t ir = 0; ir < rmesh.shape()[0]; ++ir) {
      auto r = rmesh(ir);
      for (size_t ik = 0; ik < nk; ++ik) {
        auto   k       = dyson_solver.bz_utils().mesh()(ik);
        double rk      = std::inner_product(r.begin(), r.end(), k.begin(), 0.0);
        exp_rk(ir, ik) = std::exp(std::complex<double>(0, 2 * rk * M_PI));
      }
    }
    for (size_t ik_hs = 0; ik_hs < hs_nk; ++ik_hs) {
      auto k = kmesh_hs(ik_hs);
      for (size_t ir = 0; ir < rmesh.shape()[0]; ++ir) {
        auto   r          = rmesh(ir);
        double rk         = std::inner_product(r.begin(), r.end(), k.begin(), 0.0);
        exp_kr(ik_hs, ir) = std::exp(std::complex<double>(0, -2 * rk * M_PI));
      }
    }

    matrix(transform) = matrix(exp_kr) * matrix(exp_rk) / double(nk);

    ztensor<4>                       Sigma_1_fbz(ns, hs_nk, nso, nso);
    ztensor<3>                       Sigma_w_fbz(hs_nk, nso, nso);
    // Compute orthogonalization transforamtion matrix
    for (int is = 0; is< ns; ++is ) {
      Sigma_1_fbz(is) << transform_to_hs(dyson_solver.bz_utils().ibz_to_full(sigma_1(is)), transform);
    }
    ztensor<4> Xk_hs(Sk_hs.shape());
    orth("symm", Sk_hs, Hk_hs, Sigma_1_fbz, Xk_hs);
    Eigen::FullPivLU<MatrixXcd> lusolver(nso, nso);
    // Interpolate G onto a new grid and perform symmetric orthogonalization
    g_omega_hs.fence();
    for (int iws = utils::context.global_rank; iws < ns * nw; iws += utils::context.global_size) {
      int iw = iws / ns;
      int is = iws % ns;
      Sigma_w.set_zero();
      dyson_solver.ft().tau_to_omega_ws(sigma_tau.object(), Sigma_w, iw, is);
      Sigma_w_fbz     << transform_to_hs(dyson_solver.bz_utils().ibz_to_full(Sigma_w), transform);
      for (int ik = 0; ik < hs_nk; ++ik) {
        auto muomega = dyson_solver.ft().wsample_fermi()(iw) * 1.0i + mu;
        matrix(G_w)  = matrix(Xk_hs(is, ik)) *
                      (muomega * matrix(Sk_hs(is, ik)) - matrix(Hk_hs(is, ik)) - matrix(Sigma_1_fbz(is, ik)) - matrix(Sigma_w_fbz(ik))) *
                      matrix(Xk_hs(is, ik));
        matrix(G_w_hs) = lusolver.compute(matrix(G_w)).inverse().eval();
        for (size_t i = 0; i < nso; ++i) {
          g_omega_hs.object()(iw, is, ik, i) = G_w_hs(i, i);
        }
      }
    }
    g_omega_hs.fence();
    MPI_Datatype dt_matrix     = utils::create_matrix_datatype<std::complex<double>>(nso * nso);
    MPI_Op       matrix_sum_op = utils::create_matrix_operation<std::complex<double>>();
    g_omega_hs.fence();
    if (!utils::context.node_rank) {
      utils::allreduce(MPI_IN_PLACE, g_omega_hs.object().data(), g_omega_hs.object().size() / (nso * nso), dt_matrix,
                       matrix_sum_op, utils::context.internode_comm);
    }
    g_omega_hs.fence();
    g_tau_hs.fence();
    if (!utils::context.node_rank) dyson_solver.ft().omega_to_tau(g_omega_hs.object(), g_tau_hs.object(), 1);
    g_tau_hs.fence();
    MPI_Type_free(&dt_matrix);
    MPI_Op_free(&matrix_sum_op);
    if (!utils::context.global_rank) {
      h5pp::archive res(results_file, "w");
      res["G_tau_hs/data"] << g_tau_hs.object();
      res["G_tau_hs/mesh"] << dyson_solver.ft().sd().repn_fermi().tsample();
      res["Sigma_1_hs"] << Sigma_1_fbz;
      res["Hk_hs"] << Hk_hs;
      res["Sk_hs"] << Sk_hs;
      res.close();
    }
    MPI_Barrier(utils::context.global);
  }
}

#endif  // MBPT_KINTER_H
