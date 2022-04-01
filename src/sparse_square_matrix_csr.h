#ifndef _SPARSE_SQUARE_MATRIX_CSR_H
#define _SPARSE_SQUARE_MATRIX_CSR_H

#include "device_buffer.h"
#include "cuda_solver.h"

namespace cuba
{

template <typename T>
class SparseSquareMatrixCSR
{
public:

	SparseSquareMatrixCSR() : size_(0), nnz_(0) {}

	void resize(int size)
	{
		size_ = size;
		rowPtr_.resize(size + 1);
	}

	void resizeNonZeros(int nnz)
	{
		nnz_ = nnz;
		values_.resize(nnz);
		colInd_.resize(nnz);
	}

	void upload(const T* values = nullptr, const int* rowPtr = nullptr, const int* colInd = nullptr)
	{
		if (values)
			values_.upload(values);
		if (rowPtr)
			rowPtr_.upload(rowPtr);
		if (colInd)
			colInd_.upload(colInd);
	}

	void download(T* values = nullptr, int* rowPtr = nullptr, int* colInd = nullptr) const
	{
		if (values)
			values_.download(values);
		if (rowPtr)
			rowPtr_.download(rowPtr);
		if (colInd)
			colInd_.download(colInd);
	}

	T* val() { return values_.data(); }
	int* rowPtr() { return rowPtr_.data(); }
	int* colInd() { return colInd_.data(); }

	const T* val() const { return values_.data(); }
	const int* rowPtr() const { return rowPtr_.data(); }
	const int* colInd() const { return colInd_.data(); }

	int size() const { return size_; }
	int nnz() const { return nnz_; }

	cusparseMatDescr_t desc() const { return desc_; }

private:

	DeviceBuffer<T> values_;
	DeviceBuffer<int> rowPtr_;
	DeviceBuffer<int> colInd_;
	int size_, nnz_;
	CusparseMatDescriptor desc_;
};


}

#endif // _SPARSE_SQUARE_MATRIX_CSR_H