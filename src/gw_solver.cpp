/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#include "green/mbpt/gw_solver.h"

#include <unsupported/Eigen/MatrixFunctions>

#include "green/mbpt/common_utils.h"

namespace green::mbpt {

  void gw_solver::solve(G_type& g, S1_type&, St_type& sigma_tau) {
    _kernel->solve(g, sigma_tau);
  }

}  // namespace green::mbpt
