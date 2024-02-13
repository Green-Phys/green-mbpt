/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#include "green/mbpt/hf_solver.h"

namespace green::mbpt {
  void hf_solver::solve(utils::shared_object<ztensor<5>>& g, ztensor<4>& sigma1, utils::shared_object<ztensor<5>>&) {
    std::array<size_t, 4> shape;
    std::copy(g.object().shape().begin() + 1, g.object().shape().end(), shape.begin());
    ztensor<4> dm(shape);
    dm << g.object()(g.object().shape()[0] - 1);
    dm *= _spin_prefactor;
    sigma1 << _kernel->solve(dm);
  }
}  // namespace green::mbpt
