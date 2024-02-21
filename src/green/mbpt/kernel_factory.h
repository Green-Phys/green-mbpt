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

#ifndef GREEN_MBPT_KERNEL_FACTORY_H
#define GREEN_MBPT_KERNEL_FACTORY_H

#include "kernels.h"

#ifdef GREEN_CUSTOM_KERNEL_HEADER
#include GREEN_CUSTOM_KERNEL_HEADER
#endif

namespace green::mbpt::kernels {
  class hf_kernel_factory {
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;
    using x_type     = ztensor<4>;

  public:
    static std::tuple<std::shared_ptr<void>, std::function<x_type(const x_type&)>> get_kernel(bool X2C, const params::params& p,
                                                                                              size_t nao, size_t nso, size_t ns,
                                                                                              size_t NQ, double madelung,
                                                                                              const bz_utils_t& bz_utils,
                                                                                              const ztensor<4>& S_k) {
      if (p["kernel"].as<kernel_type>() == CPU) {
        if (X2C) {
          std::shared_ptr<void> kernel(new hf_x2c_cpu_kernel(p, nao, nso, ns, NQ, madelung, bz_utils, S_k));
          std::function         callback = [kernel](const x_type& dm) -> x_type {
            return static_cast<hf_x2c_cpu_kernel*>(kernel.get())->solve(dm);
          };
          return std::tuple{kernel, callback};
        }
        std::shared_ptr<void> kernel(new hf_scalar_cpu_kernel(p, nao, nso, ns, NQ, madelung, bz_utils, S_k));
        std::function         callback = [kernel](const x_type& dm) -> x_type {
          return static_cast<hf_scalar_cpu_kernel*>(kernel.get())->solve(dm);
        };
        return std::tuple{kernel, callback};
      }
#ifdef GREEN_CUSTOM_KERNEL_HEADER
      if (p["kernel"].as<kernel_type>() == GREEN_CUSTOM_KERNEL_ENUM ) {
        return custom_hf_kernel(X2C, p, nao, nso, ns, NQ, madelung, bz_utils, S_k);
      }
#endif
      throw mbpt_kernel_error("Cannot determine HF kernel");
    }
  };

  class gw_kernel_factory {
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;
    using G_type     = utils::shared_object<ztensor<5>>;

  public:
    static std::tuple<std::shared_ptr<void>, std::function<void(G_type&, G_type&)>> get_kernel(
        bool X2C, const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, const grids::transformer_t& ft,
        const bz_utils_t& bz_utils, const ztensor<4>& S_k) {
      if (p["kernel"].as<kernel_type>() == CPU) {
        std::shared_ptr<void> kernel(new gw_cpu_kernel(p, nao, nso, ns, NQ, ft, bz_utils, S_k, X2C));
        std::function callback = [kernel](G_type& g, G_type& s) { static_cast<gw_cpu_kernel*>(kernel.get())->solve(g, s); };
        return std::tuple{kernel, callback};
      }
#ifdef GREEN_CUSTOM_KERNEL_HEADER
      if (p["kernel"].as<kernel_type>() == GREEN_CUSTOM_KERNEL_ENUM ) {
        return custom_gw_kernel(X2C, p, nao, nso, ns, NQ, ft, bz_utils, S_k);
      }
#endif
      throw mbpt_kernel_error("Cannot determine GW kernel");
    }
  };
}  // namespace green::mbpt::kernels

#endif  // GREEN_MBPT_KERNEL_FACTORY_H
