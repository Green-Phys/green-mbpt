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

#include "common_defs.h"
#include "df_integral_t.h"

namespace green::mbpt {

  class hf_solver {
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;

  public:
    hf_solver(const params::params& p, const bz_utils_t& bz_utils, const ztensor<4>& S_k) :
        _nk(bz_utils.nk()), _ink(bz_utils.ink()), _X2C(false), _S_k(S_k), _bz_utils(bz_utils), _hf_path(p["dfintegral_hf_file"]) {
      h5pp::archive ar(p["input_file"]);
      ar["params/NQ"] >> _NQ;
      ar["params/nao"] >> _nao;
      ar["params/nso"] >> _nso;
      ar["params/ns"] >> _ns;
      ar["HF/madelung"] >> _madelung;
      ar.close();
      _X2C = _nso != _nao;
      if (_ns != 1 and _X2C) {
        throw std::logic_error("For GSCF methods, \"ns\" has to be 1.");
      }
    }

    void solve(utils::shared_object<ztensor<5>>& G, ztensor<4>& Sigma1, utils::shared_object<ztensor<5>>& Sigma_tau);
    /**
     * Dummy function. Will be overwritten in cuda solver at high cpu memory mode
     */
    void set_shared_Coulomb() {}

  protected:
    ztensor<4> solve_HF_scalar(const ztensor<4>& dm);
    // functions for 2c HF. Will merge to scalar HF later
    ztensor<4> solve_HF_2c(const ztensor<4>& dm);
    MatrixXcd  compute_exchange(int ik, ztensor<3>& dm_s1_s2, ztensor<3>& dm_ms1_ms2, ztensor<3>& v, df_integral_t& coul_int1,
                                ztensor<3>& Y, MMatrixXcd& Ym, MMatrixXcd& Ymm, ztensor<3>& Y1, MMatrixXcd& Y1m, MMatrixXcd& Y1mm,
                                MMatrixXcd& vmm, ztensor<3>& v2, MMatrixXcd& v2m, MMatrixXcd& v2mm);
    MatrixXcd  compute_exchange_ab(int ik, ztensor<3>& dm_ab, ztensor<3>& v, df_integral_t& coul_int1, ztensor<3>& Y,
                                   MMatrixXcd& Ym, MMatrixXcd& Ymm, ztensor<3>& Y1, MMatrixXcd& Y1m, MMatrixXcd& Y1mm,
                                   MMatrixXcd& vmm, ztensor<3>& v2, MMatrixXcd& v2m, MMatrixXcd& v2mm);

    bool       _X2C;
    // number of atomic orbitals per cell
    size_t     _nao;
    size_t     _nso;
    // number of cells for GF2 loop
    size_t     _nk;
    // number of k-point after time-reversal symmetry
    size_t     _ink;
    // number of spins
    size_t     _ns;
    // auxiliraly basis size
    size_t     _NQ;
    // madelung constant
    double     _madelung;
    // overlap
    const ztensor<4>& _S_k;

    const bz_utils_t& _bz_utils;

    const std::string _hf_path;
  };

}  // namespace green::mbpt

#endif  // MPIGF2_GF2SOLVER_H
