/*
 * Copyright (c) 2023 University of Michigan
 *
 */
#ifndef MBPT_COMMON_UTILS_H
#define MBPT_COMMON_UTILS_H

#include <green/utils/timing.h>
#include <green/grids.h>
#include "df_integral_t.h"

namespace green::mbpt {

  inline void print_leakage(double leakage, const std::string& object) {
    std::cout << "Leakage of " + object << ": " << leakage << std::endl;
    if (leakage > 1e-8) std::cerr << "Warning: The leakage is larger than 1e-8" << std::endl;
  }

  inline auto compute_energy(const ztensor<5>& g_tau, const ztensor<4>& sigma1, const ztensor<5>& sigma_tau,
                             const ztensor<4>& H_k, const grids::transformer_t& ft,
                             const symmetry::brillouin_zone_utils<symmetry::inv_symm_op>& bz, bool X2C) {
    size_t     _nso = g_tau.shape()[4];
    size_t     _ns  = g_tau.shape()[1];
    size_t     _ink = g_tau.shape()[2];
    size_t     _nw  = ft.sd().repn_fermi().nw();
    size_t     _nts = ft.sd().repn_fermi().nts();
    // Transform G and Sigma to Matsubara axis
    ztensor<2> G_w(_nso, _nso);
    ztensor<2> Sigma_w(_nso, _nso);
    ztensor<4> dmr(_ns, _ink, _nso, _nso);
    dmr << g_tau(_nts - 1);
    dmr *= (_ns == 2 or X2C) ? -1.0 : -2.0;

    MatrixXcd GS_w = MatrixXcd::Zero(_nw, 1);
    MatrixXcd GS_t(1, 1);
    MatrixXcd TtBn   = ft.Ttn().block(_nts - 1, 0, 1, _nw);

    double    energy = 0.0;
    double    ehf    = 0.0;
    double    e1e    = 0.0;
    for (size_t iwsk = utils::context.global_rank; iwsk < _nw * _ns * _ink; iwsk += utils::context.global_size) {
      size_t iw = iwsk / (_ns * _ink);
      size_t is = (iwsk % (_ns * _ink)) / _ink;
      size_t ik = iwsk % _ink;
      ft.tau_to_omega_wsk(sigma_tau, Sigma_w, iw, is, ik, 1);
      ft.tau_to_omega_wsk(g_tau, G_w, iw, is, ik, 1);
      size_t k_ir = bz.symmetry().full_point(ik);
      GS_w(iw, 0) += bz.symmetry().weight()[k_ir] * (matrix(G_w) * matrix(Sigma_w)).eval().trace();
    }
    GS_t                    = TtBn * GS_w;
    double energy_prefactor = (_ns == 1 and !X2C) ? 1.0 : 0.5;
    energy                  = -GS_t(0, 0).real() * energy_prefactor;
    MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE_PRECISION, MPI_SUM, utils::context.global);
    for (size_t isk = utils::context.global_rank; isk < _ns * _ink; isk += utils::context.global_size) {
      size_t is   = isk / _ink;
      size_t ik   = isk % _ink;
      size_t k_ir = bz.symmetry().full_point(ik);
      ehf += 0.5 * (matrix(dmr(is, ik)) * (matrix(sigma1(is, ik)) + 2.0 * matrix(H_k(is, ik)))).trace().real() *
             bz.symmetry().weight()[k_ir];
      e1e += (matrix(dmr(is, ik)) * matrix(H_k(is, ik))).trace().real() * bz.symmetry().weight()[k_ir];
    }
    MPI_Allreduce(MPI_IN_PLACE, &e1e, 1, MPI_DOUBLE_PRECISION, MPI_SUM, utils::context.global);
    MPI_Allreduce(MPI_IN_PLACE, &ehf, 1, MPI_DOUBLE_PRECISION, MPI_SUM, utils::context.global);
    e1e *= bz.nkpw();
    ehf *= bz.nkpw();
    energy *= bz.nkpw();

    return std::array<double, 3>{e1e, ehf, energy};
  }
}  // namespace green::mbpt
#endif  // MBPT_COMMON_UTILS_H
