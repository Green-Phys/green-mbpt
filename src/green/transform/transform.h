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

#ifndef GREEN_TRANSFORM_TRANSFORM_H
#define GREEN_TRANSFORM_TRANSFORM_H

#include <green/h5pp/archive.h>
#include <green/ndarray/ndarray_math.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <mpi.h>

#include <Eigen/Core>

namespace green::transform {
  template <typename prec, size_t Dim>
  using tensor = ndarray::ndarray<prec, Dim>;
  template <size_t Dim>
  using ztensor = ndarray::ndarray<std::complex<double>, Dim>;
  template <size_t Dim>
  using dtensor     = ndarray::ndarray<double, Dim>;

  using MatrixXcd   = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXd    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MMatrixXcd  = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MMatrixXd   = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXcd = Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXd  = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using column      = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1, Eigen::ColMajor>;

  struct int_transform {
    std::string input_file;
    std::string in_file;
    std::string in_int_file;
    std::string out_int_file;
    int         transform;
  };

  class int_transformer {
  public:
    int_transformer(const int_transform& params) : _params(params) {
      int myid;
      MPI_Comm_rank(MPI_COMM_WORLD, &myid);
      dtensor<2>  kgrid(0ul, 0ul);
      std::string grid     = "/grid/k_mesh";

      std::string basename = _params.in_int_file;
      std::string meta     = basename + "/meta.h5";
      h5pp::archive       meta_file(meta, "r");
      meta_file["chunk_size"] >>_chunk_size;
      meta_file.close();

      h5pp::archive ar(_params.in_file, "r");
      ar["/grid/num_kpair_stored"] >> _num_kpair_stored;
      ar[grid] >> kgrid;
      _nkpts = kgrid.shape()[0];
      _kpair_irre_list.resize(_num_kpair_stored);
      _conj_kpair_list.resize(_nkpts * (_nkpts + 1) / 2);
      _trans_kpair_list.resize(_nkpts * (_nkpts + 1) / 2);
      ar["/grid/conj_pairs_list"] >> _conj_kpair_list;
      ar["/grid/trans_pairs_list"] >> _trans_kpair_list;
      ar["/grid/kpair_irre_list"] >> _kpair_irre_list;
      _nchunks = std::ceil(double(_num_kpair_stored) / _chunk_size);
      _q_ind.resize(_nkpts, _nkpts);
      _q_ind2.resize(_nkpts, _nkpts);
      get_mom_cons(ar, _nkpts);

      if (myid == 0) {
        std::cout << "nkpts = " << _nkpts << std::endl;
        std::cout << "Number of kpair stored = " << _num_kpair_stored << std::endl;
        std::cout << "Chunk size = " << _chunk_size << std::endl;
        std::cout << "Number of chunks = " << _nchunks << std::endl;
      }

      ar.close();
    }
    void transform_3point();

  private:
    const int_transform& _params;
    int                  _chunk_size;
    int                  _num_kpair_stored;
    int                  _nkpts;
    int                  _nchunks;
    std::vector<int>     _conj_kpair_list;
    std::vector<int>     _trans_kpair_list;
    std::vector<int>     _kpair_irre_list;

    tensor<int, 2>       _q_ind;
    tensor<int, 2>       _q_ind2;

    int                  find_pos(const tensor<double, 1>& k, const tensor<double, 2>& kmesh);

    tensor<double, 1>    wrap(const tensor<double, 1>& k);

    void                 get_mom_cons(h5pp::archive& file, int nk);

    int                  mom_cons(int k1, int k2, int k3) const;

    // utilities for inverse k-symmetry
    void                 read_integrals( h5pp::archive& file, int current_chunk, int chunk_size, ztensor<4>& vij_Q);

    int                  irre_pos_kpair(int idx, std::vector<int>& kpair_irre_list);

    void                 get_ki_kj(const int kikj, int& ki, int& kj, int nkpts);

    int                  get_idx_red(int k1, int k2, std::vector<int>& conj_kpair_list, std::vector<int>& trans_kpair_list,
                                     std::vector<int>& kpair_irre_list);

    int                  get_vtype(int k1, int k2, std::vector<int>& conj_kpair_list, std::vector<int>& trans_kpair_list);

    void                 symmetrization(ztensor<3>& v, int v_type);

    /**
     *
     * @param myid - current MPI proc id
     * @param basename - base path to integral files
     * @param X_k - Basis transformation matrix
     * @param UU - Impurity orbitals projection matrix
     * @param i - current impurity number
     * @param nao - number of orbitals in the unit cell
     * @param nno - number of orbitals for current impurity
     * @param NQ - size of auxiliary basis
     */
    void orthogonalize_Vij_Q(int myid, const std::string& basename, ztensor<3>& X_k, dtensor<2>& UU, int i, int nao, int nno,
                             int NQ);

    void compute_local_VijQ(int myid, int nprocs, int NQ, int nno, h5pp::archive& int_file, ztensor<4>& VijQ1, ztensor<3>& VijQ_loc);

    void extract_impurity_interaction(int myid, int nprocs,  h5pp::archive& int_file, ztensor<4>& VijQ1, ztensor<4>& VijQ2,
                                      dtensor<4>& dERI, ztensor<4>& zERI, int nno, int NQ);
  };
}  // namespace green::transform
#endif  // GREEN_TRANSFORM_TRANSFORM_H
