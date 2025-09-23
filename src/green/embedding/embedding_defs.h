/*
 * Copyright (c) 2024 University of Michigan
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

#ifndef GREEN_EMBEDDING_DEFS_H
#define GREEN_EMBEDDING_DEFS_H

#include <array>
#include <cstddef>

#include <green/params/params.h>

namespace green::embedding {

  template <typename T, size_t D>
    std::array<size_t, D + 1> operator+(const std::array<size_t, D>& a, T b) {
    std::array<size_t, D + 1> result;
    std::copy(a.begin(), a.end(), result.begin());
    result[D] = size_t(b);
    return result;
  }

  template <typename T, size_t D>
  std::array<size_t, D + 1> operator+(T b, const std::array<size_t, D>& a) {
    std::array<size_t, D + 1> result;
    std::copy(a.begin(), a.end(), result.begin() + 1);
    result[0] = size_t(b);
    return result;
  }

  enum embedding_type {
    SEET, FSC_SEET
  };

  inline void define_parameters(params::params& p) {
    p.define<embedding_type>("embedding_type", "Type of Embedding theory (SEET - inner loop self-consistency self-energy embedding; FSC_SEET - fully self-consistent self-energy embedding).");
    p.define<std::string>("seet_input", "Input file with SEET transformations.", "transform.h5");
    p.define<std::string>("weak_results", "Results from the weak-coupling simulation.");
    p.define<std::string>("bath_file", "Input file with initial bath parameters.", "bath.txt");
    p.define<std::string>("impurity_solver", "Type of the impurity solver.", "ED");
    p.define<std::string>("impurity_solver_exec", "Path to an impurity solver executable.");
    p.define<std::string>("impurity_solver_params", "Impurity solver parameters.");
    p.define<std::string>("dc_data_prefix", "Prefix for the path to the double counting data.");
    p.define<std::string>("seet_root_dir", "Directory to put output for impurity solvers.", "");
    p.define<bool>("spin_symm", "Apply spin symmetrization to hybridization function", false);
  }
}
#endif //GREEN_EMBEDDING_DEFS_H
