/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef GREEN_GW_SOLVER_T_H
#define GREEN_GW_SOLVER_T_H

#include <green/grids/transformer_t.h>
#include <green/ndarray/ndarray.h>
#include <green/ndarray/ndarray_math.h>
#include <green/params/params.h>
#include <green/symmetry/symmetry.h>
#include <green/utils/mpi_shared.h>
#include <green/utils/mpi_utils.h>
#include <green/utils/timing.h>
#include <mpi.h>

// #include "gscf/gscf_cuhf_solver_t.h"
// #include "transformer_t.h"
#include "common_defs.h"
#include "df_integral_t.h"
#include "mbpt_q0_utils_t.h"

#include "kernels.h"

namespace green::mbpt {
  /**
   * @brief GWSolver class performs self-energy calculation by means of GW approximation using density fitting
   */
  class gw_solver {
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
    gw_solver(const params::params& p, const grids::transformer_t& ft, const bz_utils_t& bz_utils, const ztensor<4>& S_k) {
      h5pp::archive ar(p["input_file"]);
      size_t nao, nso, ns, NQ;
      bool X2C;
      ar["params/nao"] >> nao;
      ar["params/nso"] >> nso;
      ar["params/ns"] >> ns;
      ar["params/NQ"] >> NQ;
      ar.close();
      X2C = nao != nso;
      _kernel = kernels::gw_kernel_factory::get_kernel(X2C, p, nao, nso, ns, NQ, ft, bz_utils, S_k);
    }

    /**
     * Solve GW equations for Self-energy
     */
    void solve(G_type& g, S1_type&, St_type& sigma_tau);

  private:
    std::unique_ptr<kernels::gw_kernel> _kernel;
  };

}  // namespace green::mbpt

#endif  // GF2_GW_SOLVER_T_H
