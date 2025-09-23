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

#include "common_defs.h"
#include "df_integral_t.h"
#include "kernel_factory.h"
#include "kernels.h"
#include "mbpt_q0_utils_t.h"

namespace green::mbpt {
  /**
   * @brief GWSolver class performs self-energy calculation by means of GW approximation.
   * Solver will use contraction scheme chosen by "KERNEL" parameter
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
     * @param bz_utils    -- Brillouin zone utilities
     * @param S_k         -- Overlap matrix
     */
    gw_solver(const params::params& p, const grids::transformer_t& ft, const bz_utils_t& bz_utils, const ztensor<4>& S_k) {
      h5pp::archive ar(p["input_file"]);
      size_t        nao, nso, ns, NQ;
      bool          X2C;
      ar["params/nao"] >> nao;
      ar["params/nso"] >> nso;
      ar["params/ns"] >> ns;
      ar["params/NQ"] >> NQ;
      ar.close();
      X2C                          = nao != nso;
      std::tie(_kernel, _callback) = kernels::gw_kernel_factory::get_kernel(X2C, p, nao, nso, ns, NQ, ft, bz_utils, S_k);
    }

    /**
     * Solve GW equations for Self-energy. This method calls GW implementation from the selected kernel.
     *
     * @param g Green's function object
     * @param sigma_tau Imaginary-time self-energy
     */
    void solve(G_type& g, S1_type&, St_type& sigma_tau);

  private:
    std::shared_ptr<void>                              _kernel;
    std::function<void(G_type& g, St_type& sigma_tau)> _callback;
  };

}  // namespace green::mbpt

#endif  // GF2_GW_SOLVER_T_H
