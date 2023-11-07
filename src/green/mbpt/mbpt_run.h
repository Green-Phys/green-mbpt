/*
 * Copyright (c) 2023 University of Michigan
 *
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

  inline void run(sc::sc_loop<shared_mem_dyson>& sc, const params::params& p) {
    scf_type         type = p["scf_type"];
    // initialize Dyson solver
    shared_mem_dyson dyson(p);
    // Allocate working arrays
    auto G_tau     = utils::shared_object<ztensor<5>>(dyson.ft().sd().repn_fermi().nts(), dyson.ns(), dyson.bz_utils().ink(),
                                                  dyson.nso(), dyson.nso());
    auto Sigma_tau = utils::shared_object<ztensor<5>>(dyson.ft().sd().repn_fermi().nts(), dyson.ns(), dyson.bz_utils().ink(),
                                                      dyson.nso(), dyson.nso());
    auto Sigma1    = ztensor<4>(dyson.ns(), dyson.bz_utils().ink(), dyson.nso(), dyson.nso());
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

}  // namespace green::mbpt

#endif  // MBPT_MBPT_RUN_H
