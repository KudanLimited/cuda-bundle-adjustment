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

#pragma once

#include "cholesky.h"
#include "scalar.h"
#include "sparse_block_matrix.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <memory>

namespace cugo
{
class LinearSolver
{
public:
    virtual bool solve(const Scalar* d_A, const Scalar* d_b, Scalar* d_x) = 0;

    virtual ~LinearSolver() {}
};

class HscSparseLinearSolver : public LinearSolver
{
public:
    using SparseMatrixCSR = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
    using PermutationMatrix = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;
    using Cholesky = CuSparseCholeskySolver<Scalar>;

    void initialize(HschurSparseBlockMatrix& Hsc);

    bool solve(const Scalar* d_A, const Scalar* d_b, Scalar* d_x) override;

private:
    std::vector<int> P_;
    Cholesky cholesky_;
};


class HppSparseLinearSolver : public LinearSolver
{
public:
    using Cholesky = CuSparseCholeskySolver<Scalar>;

    void initialize(HppSparseBlockMatrix& Hpp);

    bool solve(const Scalar* d_A, const Scalar* d_b, Scalar* d_x) override;

private:
    std::vector<int> P_;
    Cholesky cholesky_;
};

class DenseLinearSolver : public LinearSolver
{
public:
    using Cholesky = CuDenseCholeskySolver<Scalar>;

    void initialize(HppSparseBlockMatrix& Hpp);

    bool solve(const Scalar* d_A, const Scalar* d_b, Scalar* d_x) override;

private:
    Cholesky cholesky_;
};

} // namespace cugo

