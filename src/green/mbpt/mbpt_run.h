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
#include "kinter.h"

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
        sc.solve(hf, dyson.H_k(), dyson.S_k(), G_tau, Sigma1, Sigma_tau);
        break;
      }
      case GW: {
        gw_solver              gw(p, dyson.ft(), dyson.bz_utils(), dyson.S_k());
        sc::composition_solver cs(hf, gw);
        sc.solve(cs, dyson.H_k(), dyson.S_k(), G_tau, Sigma1, Sigma_tau);
        break;
      }
      case GF2: {
        gf2_solver             gf2(p, dyson.ft(), dyson.bz_utils());
        sc::composition_solver cs(hf, gf2);
        sc.solve(cs, dyson.H_k(), dyson.S_k(), G_tau, Sigma1, Sigma_tau);
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
      sc::read_results(dyson.mu(), g0_tau, sigma1, sigma_tau, p["results_file"]);
      wannier_interpolation(dyson, sigma1, sigma_tau, input, p["high_symmetry_output_file"]);
    }
    input.close();
  }

  inline void check_input(const params::params&p) {
    std::string path = p["input_file"];
    h5pp::archive ar(path, "r");
    if(ar.has_attribute("__green_version__")) {
      std::string int_version = ar.get_attribute<std::string>("__green_version__");
      if (int_version.rfind(INPUT_VERSION, 0) != 0) {
        throw mbpt_outdated_input("Input file at '" + path +"' is outdated, please run migration script python/migrate.py");
      }
    } else {
      throw mbpt_outdated_input("Input file at '" + path +"' is outdated, please run migration script python/migrate.py");
    }
    ar.close();
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
