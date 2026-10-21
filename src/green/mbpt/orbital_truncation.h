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

#ifndef GREEN_MBPT_ORBITAL_TRUNCATION_H
#define GREEN_MBPT_ORBITAL_TRUNCATION_H

#include <green/h5pp/archive.h>
#include <green/params/params.h>

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "common_defs.h"
#include "except.h"

namespace green::mbpt {

  /**
   * Orbital / auxiliary-basis truncation used by the correlated self-energy (GW only for now).
   *  - frozen core : the first `ncore` entries of `orb_reordering` are excluded
   *  - FNO         : the last `nv_del` entries of `orb_reordering` are excluded
   *  - NAF         : the last `NQ_del` auxiliary functions are excluded
   */
  struct orbital_truncation_t {
    bool                frozen_core = false;
    size_t              ncore       = 0;
    size_t              nv_del      = 0;
    size_t              NQ_del      = 0;
    size_t              nao_eff     = 0;  // nao - ncore - nv_del : orbitals kept
    size_t              NQ_eff      = 0;  // NQ - NQ_del          : aux functions kept
    std::vector<size_t> core_idx;         // AO indices of the frozen core orbitals
    std::vector<size_t> valence_idx;      // AO indices kept (size nao_eff), sorted ascending

    bool orbitals_truncated() const { return ncore > 0 || nv_del > 0; }
    bool aux_truncated() const { return NQ_del > 0; }
  };

  /**
   * Read and validate the truncation parameters.
   *  - `ncore` is only used when frozen_core is true; -1 reads params/ncore from the input file.
   *  - `orb_reordering` is used whenever orbitals are removed (frozen_core or nv_del > 0); [-1] reads
   *    params/orb_reordering from the input file. With frozen_core it is required; with FNO only
   *    (nv_del > 0, frozen_core false) the stored orbital order is used if the input file has none.
   */
  inline orbital_truncation_t read_orbital_truncation(const params::params& p, size_t nao, size_t NQ) {
    orbital_truncation_t t;
    t.frozen_core = p["frozen_core"].as<bool>();
    t.nv_del      = p["nv_del"].as<size_t>();
    t.NQ_del      = p["NQ_del"].as<size_t>();

    // Default: identity ordering, no core
    std::vector<int> orb_reordering(nao);
    std::iota(orb_reordering.begin(), orb_reordering.end(), 0);
    int ncore = 0;

    if (t.frozen_core || t.nv_del > 0) {
      std::vector<int> orb_reordering_p    = p["orb_reordering"].as<std::vector<int>>();
      // orb_reordering == [-1]: take it from the input file
      bool             need_orb_reordering = (orb_reordering_p == std::vector<int>{-1});
      // ncore == -1: take it from the input file (frozen core only)
      if (t.frozen_core) ncore = p["ncore"].as<int>();
      bool need_ncore = t.frozen_core && (ncore == -1);
      if (need_ncore || need_orb_reordering) {
        std::string   fname = p["input_file"];
        h5pp::archive ar(fname);
        try {
          if (need_ncore) ar["params/ncore"] >> ncore;
          // FNO only: a missing params/orb_reordering means "use the stored orbital order"
          if (need_orb_reordering && (t.frozen_core || ar.is_data("params/orb_reordering")))
            ar["params/orb_reordering"] >> orb_reordering;
        } catch (const std::exception& e) {
          ar.close();
          throw mbpt_invalid_truncation("'params/ncore' or 'params/orb_reordering' could not be read from " +
                                        fname + " (" + e.what() + "). Provide them in the input file or via --ncore / --orb_reordering.");
        }
        ar.close();
      }
      if (!need_orb_reordering) orb_reordering = orb_reordering_p;
    }

    // ---- validation ----
    if (ncore < 0) throw mbpt_invalid_truncation("ncore must be >= 0, got " + std::to_string(ncore));
    t.ncore = static_cast<size_t>(ncore);
    if (t.ncore + t.nv_del >= nao)
      throw mbpt_invalid_truncation("ncore + nv_del (" + std::to_string(t.ncore + t.nv_del) + ") must be < nao (" +
                                    std::to_string(nao) + ")");
    if (t.NQ_del >= NQ)
      throw mbpt_invalid_truncation("NQ_del (" + std::to_string(t.NQ_del) + ") must be < NQ (" + std::to_string(NQ) + ")");
    if (orb_reordering.size() != nao)
      throw mbpt_invalid_truncation("orb_reordering has " + std::to_string(orb_reordering.size()) + " entries, expected nao = " +
                                    std::to_string(nao));
    {
      std::vector<int> sorted = orb_reordering;
      std::sort(sorted.begin(), sorted.end());
      for (size_t i = 0; i < nao; ++i)
        if (sorted[i] != static_cast<int>(i)) throw mbpt_invalid_truncation("orb_reordering is not a permutation of 0..nao-1");
    }

    // ---- derived quantities ----
    t.nao_eff = nao - t.ncore - t.nv_del;
    t.NQ_eff  = NQ - t.NQ_del;
    t.core_idx.assign(orb_reordering.begin(), orb_reordering.begin() + t.ncore);
    t.valence_idx.assign(orb_reordering.begin() + t.ncore, orb_reordering.begin() + t.ncore + t.nao_eff);
    // Make sure that all valence orbitals needed are given in ascending order
    // Needed for inplace slicing
    std::sort(t.valence_idx.begin(), t.valence_idx.end());
    return t;
  }

  /**
   * In-place slice of the DF integrals: packs v(iq, idx[r], idx[s]) for iq < NQ into a
   * contiguous (NQ, nao_eff, nao_eff) block at the front of v's buffer (nao_eff = idx.size()).
   * Safe because idx is sorted ascending: idx[r] >= r, so every read offset
   * iq*nao*nao + idx[r]*nao + idx[s] is >= the write offset iq*nao_eff*nao_eff + r*nao_eff + s,
   * and read offsets increase monotonically, so no source is overwritten before it is read.
   */
  template <typename prec>
  void valence_slice_coulint_inplace(size_t NQ, const std::vector<size_t>& idx, tensor<prec, 3>& v) {
    const size_t nao_eff = idx.size();
    prec*        data    = v.data();
    for (size_t iq = 0; iq < NQ; ++iq)
      for (size_t r = 0; r < nao_eff; ++r)
        for (size_t s = 0; s < nao_eff; ++s) data[iq * nao_eff * nao_eff + r * nao_eff + s] = v(iq, idx[r], idx[s]);
  }

  /**
   * In-place slice of a matrix: writes M(idx[r], idx[s]) into the top-left (nao_eff x nao_eff)
   * block of M. Safe because idx is sorted ascending: idx[r] >= r and idx[s] >= s, so element
   * (r, s) is only overwritten after every read that needs it.
   */
  template <typename prec>
  void valence_slice_matrix_inplace(const std::vector<size_t>& idx, MatrixX<prec>& M) {
    const size_t nao_eff = idx.size();
    for (size_t r = 0; r < nao_eff; ++r)
      for (size_t s = 0; s < nao_eff; ++s) M(r, s) = M(idx[r], idx[s]);
  }

}  // namespace green::mbpt
#endif  // GREEN_MBPT_ORBITAL_TRUNCATION_H
