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
    if (leakage > 1e-8) std::cerr << "WARNING: The leakage is larger than 1e-8" << std::endl;
  }

  std::array<double, 3> compute_energy(const ztensor<5>& g_tau, const ztensor<4>& sigma1, const ztensor<5>& sigma_tau,
                                       const ztensor<4>& H_k, const grids::transformer_t& ft,
                                       const symmetry::brillouin_zone_utils& bz, bool X2C);

  inline std::pair<size_t, size_t> compute_local_and_offset_node_comm(size_t size, const utils::mpi_context & cntx = utils::mpi_context::context) {
    size_t local = size / cntx.node_size;
    local += (size % cntx.node_size > cntx.node_rank) ? 1 : 0;
    size_t offset = local * cntx.node_rank +
                        ((size % cntx.node_size > cntx.node_rank) ? 0 : (size % cntx.node_size));
    return {local, offset};
  }

}  // namespace green::mbpt
#endif  // MBPT_COMMON_UTILS_H
