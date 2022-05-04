#ifndef _DENSE_SQUARE_MATRIX_H
#define _DENSE_SQUARE_MATRIX_H

#include "cuda_solver.h"
#include "device_buffer.h"

namespace cuba
{
template <typename T>
class DenseSquareMatrix
{
public:
    DenseSquareMatrix() : rows_(0), cols_(0), ld_(0) {}

    void resize(int size)
    {
        rows_ = size;
        cols_ = size;
        ld_ = size;
        values_.resize(rows_ * cols_);
    }

    void upload(const T* values) { values_.upload(values); }

    void download(T* values) const { values_.download(values); }

    T* val() { return values_.data(); }
    const T* val() const { return values_.data(); }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int ld() const { return ld_; }

    cusparseMatDescr_t desc() const { return desc_; }

private:
    DeviceBuffer<T> values_;
    int rows_, cols_, ld_;
    CuMatDescriptor desc_;
};


} // namespace cuba

#endif // _SPARSE_SQUARE_MATRIX_CSR_H