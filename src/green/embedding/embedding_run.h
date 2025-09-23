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

#ifndef GREEN_EMBEDDING_RUN_H
#define GREEN_EMBEDDING_RUN_H

#include <green/mbpt/common_defs.h>
#include <green/params/params.h>
#include <green/sc/sc_loop.h>

#include "embedding_defs.h"
#include "seet_solver.h"

namespace green::embedding {

  using dyn_type = utils::shared_object<mbpt::ztensor<5>>;
  using stat_type = mbpt::ztensor<4>;
  using dc_solver_functype   = std::function<void(std::string, int, dyn_type&, stat_type&, dyn_type&)>;
  using weak_solver_functype = std::function<void(dyn_type&, stat_type&, dyn_type&)>;

  inline void read_hartree_fock_selfenergy(const params::params&                                        p,
                                           const symmetry::brillouin_zone_utils<symmetry::inv_symm_op>& bz,
                                           sc::ztensor<4>&                                              Sigma1) {
    h5pp::archive         ar(p["input_file"]);
    std::array<size_t, 4> shape = Sigma1.shape();
    shape[1]                    = bz.nk();
    mbpt::ztensor<4> tmp(shape);
    mbpt::ztensor<4> tmp2(shape);
    ar["HF/Fock-k"] >> tmp.reshape(shape + 1).view<double>();
    ar["HF/H-k"] >> tmp2.reshape(shape + 1).view<double>();
    tmp -= tmp2;
    for (size_t is = 0; is < Sigma1.shape()[0]; ++is) {
      Sigma1(is) << bz.full_to_ibz(tmp(is));
    }
  }

  dc_solver_functype get_dc_solver(params::params& p2) {
    return [&p2](std::string dc_data_path, int imp, dyn_type& G, stat_type& Sigma1, dyn_type& Sigma_t) {
      p2["input_file"]         = dc_data_path + "." + std::to_string(imp) + "/dummy.h5";
      p2["dfintegral_file"]    = dc_data_path + "." + std::to_string(imp);
      p2["dfintegral_hf_file"] = dc_data_path + "." + std::to_string(imp);
      green::symmetry::brillouin_zone_utils bz(p2);
      green::grids::transformer_t           ft(p2);
      size_t                                nso = 2, ns = 2, nk = 1, ink = 1, nts = ft.sd().repn_fermi().nts();
      auto                 Sk   = green::sc::ztensor<4>(ns, ink, nso, nso);

      const mbpt::scf_type type = p2["scf_type"];
      switch (type) {
        case mbpt::HF: {
          green::mbpt::hf_solver hf(p2, bz, Sk);
          hf.solve(G, Sigma1, Sigma_t);
          break;
        }
        case mbpt::GW: {
          green::mbpt::hf_solver hf(p2, bz, Sk);
          green::mbpt::gw_solver gw(p2, ft, bz, Sk);
          green::sc::composition_solver<green::mbpt::hf_solver, green::mbpt::gw_solver> solver(hf, gw);
          solver.solve(G, Sigma1, Sigma_t);
          if(!G.cntx().global_rank) {
            h5pp::archive ar(p2["seet_root_dir"].as<std::string>() + "/dc." + std::to_string(imp) + ".result.h5", "w");
            ar["Sigma1/data"] << Sigma1;
            ar["Sigma_t/data"] << Sigma_t.object();
            ar.close();
          }
          MPI_Barrier(G.cntx().global);
          break;
        }
        default: {
          std::cout << "scf type not implemented." << std::endl;
          break;
        }
      }
    };
  }

  inline weak_solver_functype get_weak_solver(const params::params& p, mbpt::scf_type type,
                       mbpt::shared_mem_dyson& dyson) {
    // Hartree-Fock solver is used by all perturbation solvers.
    mbpt::hf_solver hf(p, dyson.bz_utils(), dyson.S_k());
    switch (type) {
      case mbpt::HF: {
        auto cs =  std::make_shared<sc::composition_solver<mbpt::hf_solver>>(hf);
        return [&cs](dyn_type& G, stat_type& Sigma1, dyn_type& Sigma_t) { cs->solve(G, Sigma1, Sigma_t);};
      }
      case mbpt::GW: {
        mbpt::gw_solver        gw(p, dyson.ft(), dyson.bz_utils(), dyson.S_k());
        auto cs =  std::make_shared<sc::composition_solver<mbpt::hf_solver, mbpt::gw_solver>>(hf, gw);
        return [&cs](dyn_type& G, stat_type& Sigma1, dyn_type& Sigma_t) { cs->solve(G, Sigma1, Sigma_t);};
      }
      case mbpt::GF2: {
        mbpt::gf2_solver       gf2(p, dyson.ft(), dyson.bz_utils());
        auto cs =  std::make_shared<sc::composition_solver<mbpt::hf_solver, mbpt::gf2_solver>>(hf, gf2);
        return [&cs](dyn_type& G, stat_type& Sigma1, dyn_type& Sigma_t) { cs->solve(G, Sigma1, Sigma_t);};
        //sc::composition_solver cs(hf, gf2);
      }
      default: {
        throw std::runtime_error("Unsupported weak-coupling type");
      }
    }
  }

  inline void seet_job(sc::sc_loop<mbpt::shared_mem_dyson>& sc, const params::params& p, mbpt::scf_type type,
                       mbpt::shared_mem_dyson& dyson, utils::shared_object<mbpt::ztensor<5>>& G_tau,
                       utils::shared_object<mbpt::ztensor<5>>& Sigma_tau, mbpt::ztensor<4>& Sigma1) {
    /*read_hartree_fock_selfenergy(p, dyson.bz_utils(), Sigma1);
    G_tau.fence();
    if (!utils::context.node_rank) G_tau.object().set_zero();
    G_tau.fence();
    Sigma_tau.fence();
    if (!utils::context.node_rank) Sigma_tau.object().set_zero();
    Sigma_tau.fence();*/
    params::params p2           = p;
    std::string    dc_data_path = p["dc_data_prefix"].as<std::string>();
    auto dc_solver              = get_dc_solver(p2);
    auto weak_solver            = get_weak_solver(p, type, dyson);
    seet_solver seet(p, dyson.ft(), dyson.bz_utils(), dyson.H_k(), dyson.S_k(), dyson.mu(), dc_solver, weak_solver);
    sc.solve(seet, dyson.H_k(), dyson.S_k(), G_tau, Sigma1, Sigma_tau);
  }

  inline void inner_seet_job(sc::sc_loop<mbpt::shared_mem_dyson>& sc, const params::params& p, mbpt::shared_mem_dyson& dyson,
                             utils::shared_object<mbpt::ztensor<5>>& G_tau, utils::shared_object<mbpt::ztensor<5>>& Sigma_tau,
                             mbpt::ztensor<4>& Sigma1) {
    /*read_hartree_fock_selfenergy(p, dyson.bz_utils(), Sigma1);
    G_tau.fence();
    if (!utils::context.node_rank) G_tau.object().set_zero();
    G_tau.fence();
    Sigma_tau.fence();
    if (!utils::context.node_rank) Sigma_tau.object().set_zero();
    Sigma_tau.fence();*/
    params::params         p2           = p;
    std::string            dc_data_path = p["dc_data_prefix"].as<std::string>();
    std::string            grid_file    = p["grid_file"].as<std::string>();
    auto                   dc_solver    = get_dc_solver(p2);
    auto                   seet_weak = std::make_shared<seet_inner_solver>(p);
    auto                   weak_solver = [&seet_weak](utils::shared_object<mbpt::ztensor<5>>& G, mbpt::ztensor<4>& Sigma1, 
                                                      utils::shared_object<mbpt::ztensor<5>>& Sigma_t) {
      seet_weak->solve(G, Sigma1, Sigma_t);
    };
    seet_solver            seet(p, dyson.ft(), dyson.bz_utils(), dyson.H_k(), dyson.S_k(), dyson.mu(), dc_solver, weak_solver);
    sc.solve(seet, dyson.H_k(), dyson.S_k(), G_tau, Sigma1, Sigma_tau);
  }

  inline void run(sc::sc_loop<mbpt::shared_mem_dyson>& sc, const params::params& p) {
    const mbpt::scf_type   type = p["scf_type"];
    // initialize Dyson solver
    mbpt::shared_mem_dyson dyson(p);
    // Allocate working arrays
    double _ = 0.0;
    auto G_tau = utils::shared_object<mbpt::ztensor<5>>(dyson.ft().sd().repn_fermi().nts(), dyson.ns(), dyson.bz_utils().ink(),
                                                        dyson.nso(), dyson.nso());
    auto Sigma_tau = utils::shared_object<mbpt::ztensor<5>>(dyson.ft().sd().repn_fermi().nts(), dyson.ns(),
                                                            dyson.bz_utils().ink(), dyson.nso(), dyson.nso());
    auto Sigma1    = mbpt::ztensor<4>(dyson.ns(), dyson.bz_utils().ink(), dyson.nso(), dyson.nso());
    sc::read_results(_, G_tau, Sigma1, Sigma_tau, p["weak_results"].as<std::string>());
    auto embedding = p["embedding_type"].as<embedding_type>();
    if (embedding == SEET) {
      inner_seet_job(sc, p, dyson, G_tau, Sigma_tau, Sigma1);
    } else if (embedding == FSC_SEET) {
      seet_job(sc, p, type, dyson, G_tau, Sigma_tau, Sigma1);
    }
    if (!utils::context.global_rank) std::cout << "Completed." << std::endl;
  }

}  // namespace green::embedding

#endif  // GREEN_EMBEDDING_RUN_H
