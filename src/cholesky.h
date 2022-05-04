#ifndef _CHOLESKY_H
#define _CHOLESKY_H

#include "cuda/cuda_block_solver.h"
#include "cuda_solver.h"
#include "dense_square_matrix.h"
#include "device_buffer.h"
#include "macro.h"
#include "sparse_square_matrix_csr.h"

#include <cuda_runtime.h>

#include <type_traits>

namespace cuba
{
template <typename T>
class SparseCholesky
{
public:
    void init(cusolverSpHandle_t handle);

    void allocateBuffer(const SparseSquareMatrixCSR<T>& A);

    bool hasZeroPivot(int* position = nullptr) const;

    bool analyze(const SparseSquareMatrixCSR<T>& A);

    bool factorize(SparseSquareMatrixCSR<T>& A);

    void solve(int size, const T* b, T* x);

    void destroy();

    ~SparseCholesky();

private:
    cusolverSpHandle_t handle_;
    csrcholInfo_t info_;
    DeviceBuffer<unsigned char> buffer_;
};

template <typename T>
class CuSparseCholeskySolver
{
public:
    enum Info
    {
        SUCCESS,
        NUMERICAL_ISSUE
    };

    CuSparseCholeskySolver(int size = 0);

    void init();

    void resize(int size);

    void setPermutaion(int size, const int* P);

    void analyze(int nnz, const int* csrRowPtr, const int* csrColInd);

    void factorize(const T* d_A);

    void solve(const T* d_b, T* d_x);

    void permute(int size, const T* src, T* dst, const int* P);

    void reordering(int size, int nnz, const int* csrRowPtr, const int* csrColInd, int* P) const;

    Info info() const;

    void downloadCSR(int* csrRowPtr, int* csrColInd);

private:
    SparseSquareMatrixCSR<T> Acsr;
    DeviceBuffer<T> d_y, d_z, d_tmp;
    DeviceBuffer<int> d_P, d_PT, d_map;
    DeviceBuffer<int> d_tmpRowPtr, d_tmpColInd, d_nnzPerRow;

    CusparseHandle cusparse;
    CusparseSolverHandle cusolver;

    SparseCholesky<T> cholesky;

    std::vector<int> h_PT;

    Info information;
    bool doOrdering;
};

template <typename T>
class DenseCholesky
{
public:
    void init(cusolverDnHandle_t handle, cusparseHandle_t spHandle);

    void allocateBuffer(SparseSquareMatrixCSR<T>& A, DenseSquareMatrix<T>& B);

    void sparseToDense(const SparseSquareMatrixCSR<T>& A, DenseSquareMatrix<T>& B);

    bool factorize(DenseSquareMatrix<T>& A);

    void solve(const DenseSquareMatrix<T>& A, const T* b, T* x);

    ~DenseCholesky();

private:
    cusolverDnHandle_t dnHandle_;
    cusparseHandle_t spHandle_;

    int h_info;
    DeviceBuffer<T> buffer_, denseBuffer_;
    DeviceBuffer<int> info_, ipiv_;

    cusparseSpMatDescr_t spMatDescr;
    cusparseDnMatDescr_t dnMatDescr;
};

template <typename T>
class CuDenseCholeskySolver
{
public:
    enum Info
    {
        SUCCESS,
        NUMERICAL_ISSUE
    };

    CuDenseCholeskySolver(int size = 0);

    void init();

    void resize(int size);

    void allocate(int nnz, const int* csrRowPtr, const int* csrColInd);

    void factorize(const T* d_A);

    void solve(const T* d_b, T* d_x);

    Info info() const;

private:
    CusparseHandle cusparse;
    CuDenseSolverHandle cusolver;

    DenseSquareMatrix<T> Adense;
    SparseSquareMatrixCSR<T> Acsr;

    DenseCholesky<T> cholesky;

    Info information;
};

#include "cholesky.hpp"

} // namespace cuba

#endif // _CHOLESKY_H