#ifndef _CUDA_SOLVER_H
#define _CUDA_SOLVER_H

#include <type_traits>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolverDn.h>

namespace cuba
{

struct CusparseHandle
{
	CusparseHandle() { init(); }
	~CusparseHandle() { destroy(); }
	void init() { CHECK_CUSPARSE(cusparseCreate(&handle)); }
	void destroy() { CHECK_CUSPARSE(cusparseDestroy(handle)); }
	operator cusparseHandle_t() const { return handle; }
	CusparseHandle(const CusparseHandle&) = delete;
	CusparseHandle& operator=(const CusparseHandle&) = delete;
	cusparseHandle_t handle;
};

struct CusparseSolverHandle
{
	CusparseSolverHandle() { init(); }
	~CusparseSolverHandle() { destroy(); }
	void init() { CHECK_CUSOLVER(cusolverSpCreate(&handle)); }
	void destroy() { CHECK_CUSOLVER(cusolverSpDestroy(handle)); }
	operator cusolverSpHandle_t() const { return handle; }
	CusparseSolverHandle(const CusparseSolverHandle&) = delete;
	CusparseSolverHandle& operator=(const CusparseSolverHandle&) = delete;
	cusolverSpHandle_t handle;
};

struct CuMatDescriptor
{
	CuMatDescriptor() { init(); }
	~CuMatDescriptor() { destroy(); }

	void init()
	{
		CHECK_CUSPARSE(cusparseCreateMatDescr(&desc));
		CHECK_CUSPARSE(cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL));
		CHECK_CUSPARSE(cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO));
		CHECK_CUSPARSE(cusparseSetMatDiagType(desc, CUSPARSE_DIAG_TYPE_NON_UNIT));
	}

	void destroy() { cusparseDestroyMatDescr(desc); }
	operator cusparseMatDescr_t() const { return desc; }
	CuMatDescriptor(const CuMatDescriptor&) = delete;
	CuMatDescriptor& operator=(const CuMatDescriptor&) = delete;
	cusparseMatDescr_t desc;
};

struct CuDenseSolverHandle
{
	CuDenseSolverHandle() { init(); }
	~CuDenseSolverHandle() { destroy(); }
	void init() { CHECK_CUSOLVER(cusolverDnCreate(&handle)); }
	void destroy() { CHECK_CUSOLVER(cusolverDnDestroy(handle)); }
	operator cusolverDnHandle_t() const { return handle; }
	CuDenseSolverHandle(const CuDenseSolverHandle&) = delete;
	CuDenseSolverHandle& operator=(const CuDenseSolverHandle&) = delete;
	cusolverDnHandle_t handle;
};

}

#endif // _CUDA_SOLVER_H