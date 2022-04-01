/*
Copyright 2020 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "cuda_linear_solver.h"

#include <vector>

#include "device_buffer.h"
#include "cuda/cuda_block_solver.h"

namespace cuba
{

void HscSparseLinearSolver::initialize(HschurSparseBlockMatrix& Hsc)
{
	const int size = Hsc.rows();
	const int nnz = Hsc.nnzSymm();

	cholesky_.resize(size);

	// set permutation
	P_.resize(size);
	cholesky_.reordering(size, nnz, Hsc.rowPtr(), Hsc.colInd(), P_.data());
	cholesky_.setPermutaion(size, P_.data());

	// analyze
	cholesky_.analyze(nnz, Hsc.rowPtr(), Hsc.colInd());
}

bool HscSparseLinearSolver::solve(const Scalar* d_A, const Scalar* d_b, Scalar* d_x)
{
	cholesky_.factorize(d_A);

	if (cholesky_.info() != Cholesky::SUCCESS)
	{
		std::cerr << "factorize failed" << std::endl;
		return false;
	}

	cholesky_.solve(d_b, d_x);

	return true;
}

void HppSparseLinearSolver::initialize(HppSparseBlockMatrix& Hpp)
{
	const int size = Hpp.rows();
	const int nnz = Hpp.nnzSymm();

	cholesky_.resize(size);

	// set permutation
	P_.resize(size);
	cholesky_.reordering(size, nnz, Hpp.rowPtr(), Hpp.colInd(), P_.data());
	cholesky_.setPermutaion(size, P_.data());

	// analyze
	cholesky_.analyze(nnz, Hpp.rowPtr(), Hpp.colInd());
}

bool HppSparseLinearSolver::solve(const Scalar* d_A, const Scalar* d_b, Scalar* d_x)
{
	cholesky_.factorize(d_A);

	if (cholesky_.info() != Cholesky::SUCCESS)
	{
		std::cerr << "factorize failed" << std::endl;
		return false;
	}

	cholesky_.solve(d_b, d_x);

	return true;
}



} // namespace cuba
