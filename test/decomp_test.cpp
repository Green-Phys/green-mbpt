
#include <green/mbpt/common_defs.h>
#include <green/transform/transform.h>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>

TEST_CASE("CheckIntegralDecomposition") {
  size_t                           nao = 2;
  std::random_device               rd;
  std::mt19937                     gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  double                           tol = 1e-9;
  green::mbpt::dtensor<4>          A(nao, nao, nao, nao);
  green::mbpt::dtensor<4>          Uijkl(nao, nao, nao, nao);

  // filling Tensor
  for (int i = 0; i < nao; ++i) {
    for (int j = 0; j < nao; ++j) {
      for (int k = 0; k < nao; ++k) {
        for (int l = 0; l < nao; ++l) {
          A(i, j, k, l) = dis(gen);
        }
      }
    }
  }

  for (int i = 0; i < nao; ++i) {
    for (int j = 0; j < nao; ++j) {
      for (int k = 0; k < nao; ++k) {
        for (int l = 0; l < nao; ++l) {
          double sum = 0.0;
          for (int q1 = 0; q1 < nao; ++q1) {
            for (int q2 = 0; q2 < nao; ++q2) {
              sum += A(i, j, q1, q2) * A(k, l, q1, q2);
            }
          }
          Uijkl(i, j, k, l) = sum;
        }
      }
    }
  }

  // decompose
  auto                    VijQ = green::transform::decomp_interaction(Uijkl, tol);

  // Reassemble into Uijkl_new
  green::mbpt::dtensor<4> Uijkl_new(nao, nao, nao, nao);
  int                     nq = VijQ.shape()[2];
  for (size_t i = 0; i < nao; ++i) {
    for (size_t j = 0; j < nao; ++j) {
      for (size_t k = 0; k < nao; ++k) {
        for (size_t l = 0; l < nao; ++l) {
          for (size_t Q = 0; Q < nq; ++Q) {
            Uijkl_new(i, j, k, l) += VijQ(i, j, Q) * VijQ(k, l, Q);
          }
          // std::cout << Uijkl_new(i, j, k, l) << " " << Uijkl(i, j, k, l) << std::endl;
        }
      }
    }
  }

  // check integrals
  REQUIRE(std::equal(Uijkl.begin(), Uijkl.end(), Uijkl_new.begin(), [tol](double a, double b) { return std::abs(a - b) < tol; }));
}
