/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef GF2_MBPT_Q0_UTILS_T_H
#define GF2_MBPT_Q0_UTILS_T_H

#include <mpi.h>

#include "common_defs.h"
#include <green/grids/transformer_t.h>

namespace green::mbpt {
  // TODO Merge two different corrections into this class
  class mbpt_q0_utils_t {
  public:
    mbpt_q0_utils_t(size_t ink, size_t NQ, const ztensor<4>&S_k, const std::string & path, sigma_q0_treatment_e q0_treatment):
    _ink(ink), _S_k(S_k), _q_abs(ink), _q0_treatment(q0_treatment) {
      if (_q0_treatment == extrapolate) {
        std::string Aq_path = path + "/AqQ.h5";
        // Read _Aq, madelung constant
        if (std::filesystem::exists(Aq_path)) {
          green::h5pp::archive int_file(Aq_path, "r");
          int_file["AqQ"] >> _AqQ;
          int_file["q_abs"] >> _q_abs;
          int_file["madelung"] >> _madelung;
          int_file.close();
          _NQ = _AqQ.shape()[1];
          //check_Aq();
        } else {
          std::cout << "## Warning: " << Aq_path << " is not found! Extrapolation treatment for GW self-energy will be disabled." << std::endl;
          _q0_treatment = ignore_G0;
        }
      }
    }
    
    void resize(size_t NQ) {
      _NQ = NQ;
    }

    void aux_to_PW_00(ztensor<4> &X_aux, ztensor<2> &X_PW_00, size_t iq);
    void check_AqQ();

    void etrapolate_dielectric_inv_q0();
    std::complex<double> extrapolate_q0(std::complex<double> *eps_q_inv_wn_ptr, size_t fit_order, double q_max = 1.0, bool debug = false);
    void polyfit(double *x, double *y, size_t fit_order, size_t num_sample, MatrixXd &c);
    std::vector<int> filter_q_abs(double q_max = 1.0);

    /**
     * Apply the extrapolated GW finite-size correction
     */
    void GW_q0_correction(ztensor<2> &eps_inv_wq, ztensor_view<5> &Sigma, ztensor_view<5> &Gtau,
                          const grids::transformer_t &ft, bool X2C,
                          size_t myid, size_t intranode_rank, size_t intranode_size, MPI_Win win_Sigma);

    /**
     * Contractions of the extrapolated self-energy finite-size correction
     */
    void apply_q0_correction(ztensor<2> &eps_q0_inv_t, ztensor_view<5> &Sigma, ztensor_view<5> &G_tau,
                             size_t intranode_rank, size_t intranode_size, MPI_Win win_Sigma);
    /**
     * Two-component version of contractions
     */
    void apply_q0_correction_2C(ztensor<2> &eps_q0_inv_t, ztensor_view<5> &Sigma, ztensor_view<5> &G_tau,
                             size_t intranode_rank, size_t intranode_size, MPI_Win win_Sigma);


    sigma_q0_treatment_e _q0_treatment;
    size_t _ink;
    size_t _NQ;

    ztensor<2> _AqQ;
    const ztensor<4> &_S_k;
    std::vector<double> _q_abs;
    // Madelung constant
    double _madelung;

    double madelung() const {return _madelung;}
    sigma_q0_treatment_e q0_treatment() const {return _q0_treatment;}
    const ztensor<2> &AqQ() const {return _AqQ;}
  };
}


#endif //GF2_MBPT_Q0_UTILS_T_H
