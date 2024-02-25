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

#ifndef MBPT_MBPT_RUN_H
#define MBPT_MBPT_RUN_H

#include <green/params/params.h>
#include <green/sc/sc_loop.h>

#include "common_defs.h"
#include "dyson.h"
#include "gf2_solver.h"
#include "gw_solver.h"
#include "hf_solver.h"

namespace green::mbpt {

  inline void read_hartree_fock_selfenergy(const params::params&                                        p,
                                           const symmetry::brillouin_zone_utils<symmetry::inv_symm_op>& bz,
                                           sc::ztensor<4>&                                              Sigma1) {
    h5pp::archive         ar(p["input_file"]);
    std::array<size_t, 4> shape = Sigma1.shape();
    shape[1]                    = bz.nk();
    ztensor<4> tmp(shape);
    ztensor<4> tmp2(shape);
    ar["HF/Fock-k"] >> tmp.reshape(shape + 1).view<double>();
    ar["HF/H-k"] >> tmp2.reshape(shape + 1).view<double>();
    tmp -= tmp2;
    for (size_t is = 0; is < Sigma1.shape()[0]; ++is) {
      Sigma1(is) << bz.full_to_ibz(tmp(is));
    }
  }

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

  inline void compute_S_sqrt(const ztensor<3>& Sk, ztensor<3>& Sk_12_inv) {
    size_t ink     = Sk.shape()[0];
    size_t nso     = Sk.shape()[1];
    using Matrixcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    Eigen::SelfAdjointEigenSolver<Matrixcd> solver(nso);
    for (size_t ik = 0; ik < ink; ++ik) {
      Matrixcd S = matrix(Sk(ik));
      solver.compute(S);
      matrix(Sk_12_inv(ik)) =
          solver.eigenvectors() * (solver.eigenvalues().cwiseSqrt().asDiagonal().inverse()) * solver.eigenvectors().adjoint();
    }
  }

  template <typename Dyson>
  void wannier_interpolation(const Dyson& dyson_solver, const ztensor<4>& sigma_1,
                             const utils::shared_object<ztensor<5>>& sigma_tau, const h5pp::archive& input,
                             const std::string& results_file) {
    using tau_hs_t = utils::shared_object<ztensor<4>>;
    // Init arrays
    dtensor<2> kmesh_hs;
    dtensor<2> rmesh;
    ztensor<3> Hk_hs;
    ztensor<3> Sk_hs;
    input["high_symm_path/k_mesh"] >> kmesh_hs;
    input["high_symm_path/r_mesh"] >> rmesh;
    input["high_symm_path/Hk"] >> Hk_hs;
    input["high_symm_path/Sk"] >> Sk_hs;
    size_t     nts   = sigma_tau.object().shape()[0];
    size_t     ns    = sigma_tau.object().shape()[1];
    size_t     ink   = sigma_tau.object().shape()[2];
    size_t     nso   = sigma_tau.object().shape()[3];
    size_t     nw    = nts - 2;
    size_t     nk    = dyson_solver.bz_utils().nk();
    size_t     hs_nk = kmesh_hs.shape()[0];
    ztensor<3> Sigma_w(ink, nso, nso);
    ztensor<2> G_w(nso, nso);
    ztensor<2> G_w_hs(nso, nso);
    ztensor<2> transform(hs_nk, nk);
    ztensor<3> Sk_hs_12_inv(Sk_hs.shape());
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
    // Compute orthogonalization transforamtion matrix
    compute_S_sqrt(Sk_hs, Sk_hs_12_inv);
    Eigen::FullPivLU<MatrixXcd> lusolver(nso, nso);
    // Interpolate G onto a new grid and perform symmetric orthogonalization
    g_omega_hs.fence();
    for (int iws = utils::context.global_rank; iws < ns * nw; iws += utils::context.global_size) {
      int iw = iws / ns;
      int is = iws % ns;
      Sigma_w.set_zero();
      dyson_solver.ft().tau_to_omega_ws(sigma_tau.object(), Sigma_w, iw, is);
      auto Sigma_1_fbz = transform_to_hs(dyson_solver.bz_utils().ibz_to_full(sigma_1(is)), transform);
      auto Sigma_w_fbz = transform_to_hs(dyson_solver.bz_utils().ibz_to_full(Sigma_w), transform);
      for (int ik = 0; ik < hs_nk; ++ik) {
        auto muomega = dyson_solver.ft().wsample_fermi()(iw) * 1.0i + dyson_solver.mu();

        matrix(G_w)  = matrix(Sk_hs_12_inv(ik)) *
                      (muomega * matrix(Sk_hs(ik)) - matrix(Hk_hs(ik)) - matrix(Sigma_1_fbz(ik)) - matrix(Sigma_w_fbz(ik))) *
                      matrix(Sk_hs_12_inv(ik));
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
      res.close();
    }
    MPI_Barrier(utils::context.global);
  }

  inline void sc_job(sc::sc_loop<shared_mem_dyson>& sc, const params::params& p, scf_type type, shared_mem_dyson& dyson,
                     utils::shared_object<ztensor<5>>& G_tau, utils::shared_object<ztensor<5>>& Sigma_tau, ztensor<4>& Sigma1) {
    read_hartree_fock_selfenergy(p, dyson.bz_utils(), Sigma1);
    G_tau.fence();
    if (!utils::context.node_rank) G_tau.object().set_zero();
    G_tau.fence();
    Sigma_tau.fence();
    if (!utils::context.node_rank) Sigma_tau.object().set_zero();
    Sigma_tau.fence();
    // Hartree-Fock solver is used by all perturbation solvers.
    hf_solver hf(p, dyson.bz_utils(), dyson.S_k());
    switch (type) {
      case HF: {
        sc.solve(hf, G_tau, Sigma1, Sigma_tau);
        break;
      }
      case GW: {
        gw_solver              gw(p, dyson.ft(), dyson.bz_utils(), dyson.S_k());
        sc::composition_solver cs(hf, gw);
        sc.solve(cs, G_tau, Sigma1, Sigma_tau);
        break;
      }
      case GF2: {
        gf2_solver             gf2(p, dyson.ft(), dyson.bz_utils());
        sc::composition_solver cs(hf, gf2);
        sc.solve(cs, G_tau, Sigma1, Sigma_tau);
        break;
      }
      default: {
        break;
      }
    }
  }

  inline void winter_job(sc::sc_loop<shared_mem_dyson>& sc, const params::params& p, shared_mem_dyson& dyson,
                         utils::shared_object<ztensor<5>>& g0_tau, utils::shared_object<ztensor<5>>& sigma_tau,
                         ztensor<4>& sigma1) {
    h5pp::archive input(p["input_file"]);
    if (input.has_group("high_symm_path")) {
      if (!utils::context.global_rank) std::cout << "Running Wannier interpolation" << std::endl;
      sc::read_results(g0_tau, sigma1, sigma_tau, p["results_file"]);
      wannier_interpolation(dyson, sigma1, sigma_tau, input, p["high_symmetry_output_file"]);
    }
    input.close();
  }

  inline void run(sc::sc_loop<shared_mem_dyson>& sc, const params::params& p) {
    const auto       jobs = p["jobs"].as<std::vector<job_type>>();
    const scf_type   type = p["scf_type"];
    // initialize Dyson solver
    shared_mem_dyson dyson(p);
    // Allocate working arrays
    auto G_tau     = utils::shared_object<ztensor<5>>(dyson.ft().sd().repn_fermi().nts(), dyson.ns(), dyson.bz_utils().ink(),
                                                  dyson.nso(), dyson.nso());
    auto Sigma_tau = utils::shared_object<ztensor<5>>(dyson.ft().sd().repn_fermi().nts(), dyson.ns(), dyson.bz_utils().ink(),
                                                      dyson.nso(), dyson.nso());
    auto Sigma1    = ztensor<4>(dyson.ns(), dyson.bz_utils().ink(), dyson.nso(), dyson.nso());
    for (const auto job : jobs) {
      if (job == SC) {
        sc_job(sc, p, type, dyson, G_tau, Sigma_tau, Sigma1);
      } else if (job == WINTER) {
        winter_job(sc, p, dyson, G_tau, Sigma_tau, Sigma1);
      }
      if (!utils::context.global_rank) std::cout << "Job " << magic_enum::enum_name(job) << " is finished." << std::endl;
    }
    if (!utils::context.global_rank) std::cout << "Completed." << std::endl;
  }

}  // namespace green::mbpt

#endif  // MBPT_MBPT_RUN_H
