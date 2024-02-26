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

#ifndef GREEN_MBPT_EXCEPT_H
#define GREEN_MBPT_EXCEPT_H

#include <stdexcept>

namespace green::mbpt {
  class mbpt_kernel_error : public std::runtime_error {
  public:
    explicit mbpt_kernel_error(const std::string& what) : std::runtime_error(what) {}
  };

  class mbpt_wrong_grid : public std::runtime_error {
  public:
    explicit mbpt_wrong_grid(const std::string& what) : std::runtime_error(what) {}
  };

  class mbpt_chemical_potential_search_failure : public std::runtime_error {
  public:
    explicit mbpt_chemical_potential_search_failure(const std::string& what) : std::runtime_error(what) {}
  };
}  // namespace green::mbpt
#endif  // GREEN_MBPT_EXCEPT_H
