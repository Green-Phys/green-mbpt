/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef MPIGF2_GF2SOLVER_H
#define MPIGF2_GF2SOLVER_H

#include <green/params/params.h>
#include <green/symmetry/symmetry.h>
#include <green/utils/mpi_shared.h>
#include <green/utils/mpi_utils.h>
#include <green/utils/timing.h>

#include "common_defs.h"
#include "df_integral_t.h"
#include "kernel_factory.h"
#include "kernels.h"

namespace green::mbpt {

  class hf_solver {
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;
    using callback_t = std::function<ztensor<4>(const ztensor<4>&, const utils::mpi_context&)>;

  public:
    hf_solver(const params::params& p, const bz_utils_t& bz_utils, const ztensor<4>& S_k) {
      size_t        NQ, nao, nso, ns;
      double        madelung;
      h5pp::archive ar(p["input_file"]);
      ar["params/NQ"] >> NQ;
      ar["params/nao"] >> nao;
      ar["params/nso"] >> nso;
      ar["params/ns"] >> ns;
      ar["HF/madelung"] >> madelung;
      ar.close();
      bool X2C        = nso != nao;
      _spin_prefactor = (ns == 2 or X2C) ? -1.0 : -2.0;
      if (ns != 1 and X2C) {
        throw std::logic_error("For GSCF methods, \"ns\" has to be 1.");
      }
      std::tie(_kernel,_callback) = kernels::hf_kernel_factory::get_kernel(X2C, p, nao, nso, ns, NQ, madelung, bz_utils, S_k);
    }
    void solve(utils::shared_object<ztensor<5>>& G, ztensor<4>& Sigma1, utils::shared_object<ztensor<5>>& Sigma_tau);

  protected:
    double                _spin_prefactor;

    std::shared_ptr<void> _kernel;
    callback_t            _callback;
  };

}  // namespace green::mbpt

#endif  // MPIGF2_GF2SOLVER_H
