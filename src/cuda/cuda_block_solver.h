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

#ifndef __CUDA_BLOCK_SOLVER_H__
#define __CUDA_BLOCK_SOLVER_H__

#include "cuda/cuda_constants.h"
#include "device_matrix.h"
#include "fixed_vector.h"
#include "measurements.h"

namespace cugo
{
namespace gpu
{
template <int N>
using Vecxd = Vec<Scalar, N>;

template <int N>
using GpuVecxd = GpuVec<Vecxd<N>>;

// kernel functions
void waitForKernelCompletion();

void setCameraParameters(const Scalar* camera);

void buildHplStructure(
    GpuVec3i& blockpos, GpuHplBlockMat& Hpl, GpuVec1i& indexPL, GpuVec1i& nnzPerCol);

void findHschureMulBlockIndices(
    const GpuHplBlockMat& Hpl, const GpuHscBlockMat& Hsc, GpuVec3i& mulBlockIds);

Scalar maxDiagonal(const GpuPxPBlockVec& Hpp, Scalar* maxD);

Scalar maxDiagonal(const GpuLxLBlockVec& Hll, Scalar* maxD);

void addLambda(GpuPxPBlockVec& Hpp, Scalar lambda, GpuPx1BlockVec& backup);

void addLambda(GpuLxLBlockVec& Hll, Scalar lambda, GpuLx1BlockVec& backup);

void restoreDiagonal(GpuPxPBlockVec& Hpp, const GpuPx1BlockVec& backup);

void restoreDiagonal(GpuLxLBlockVec& Hll, const GpuLx1BlockVec& backup);

void computeBschure(
    const GpuPx1BlockVec& bp,
    const GpuHplBlockMat& Hpl,
    const GpuLxLBlockVec& Hll,
    const GpuLx1BlockVec& bl,
    GpuPx1BlockVec& bsc,
    GpuLxLBlockVec& invHll,
    GpuPxLBlockVec& Hpl_invHll);

void computeHschure(
    const GpuPxPBlockVec& Hpp,
    const GpuPxLBlockVec& Hpl_invHll,
    const GpuHplBlockMat& Hpl,
    const GpuVec3i& mulBlockIds,
    GpuHscBlockMat& Hsc);

void convertHschureBSRToCSR(
    const GpuHscBlockMat& HscBSR, const GpuVec1i& BSR2CSR, GpuVec1d& HscCSR);
void convertHppBSRToCSR(const GpuHppBlockMat& HppBSR, const GpuVec1i& BSR2CSR, GpuVec1d& HppCSR);

void twistCSR(
    int size,
    int nnz,
    const int* srcRowPtr,
    const int* srcColInd,
    const int* P,
    int* dstRowPtr,
    int* dstColInd,
    int* dstMap,
    int* nnzPerRow);

void permute(int size, const Scalar* src, Scalar* dst, const int* P);

void schurComplementPost(
    const GpuLxLBlockVec& invHll,
    const GpuLx1BlockVec& bl,
    const GpuHplBlockMat& Hpl,
    const GpuPx1BlockVec& xp,
    GpuLx1BlockVec& xl);

void updatePoses(const GpuPx1BlockVec& xp, GpuVecSe3d& estimate);

void updateLandmarks(const GpuLx1BlockVec& xl, GpuVec3d& Xws);

void computeScale(const GpuVec1d& x, const GpuVec1d& b, Scalar* scale, Scalar lambda);

template <int M>
void constructQuadraticForm_(
    const GpuVec3d& Xcs,
    const GpuVecSe3d& se3,
    GpuVecxd<M>& errors,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const GpuVec1i& edge2Hpl,
    const GpuVec1b& flags,
    GpuPxPBlockVec& Hpp,
    GpuPx1BlockVec& bp,
    GpuLxLBlockVec& Hll,
    GpuLx1BlockVec& bl,
    GpuHplBlockMat& Hpl);

template <int M>
Scalar computeActiveErrors_(
    const GpuVecSe3d& poseEstimate,
    const GpuVec3d& landmarkEstimate,
    const GpuVecxd<M>& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const Scalar errorThreshold,
    GpuVecxd<M>& errors,
    GpuVec1i& outliers,
    GpuVec3d& Xcs,
    Scalar* chi);

Scalar computeActiveErrors_DepthBa(
    const GpuVecSe3d& poseEstimate,
    const GpuVec3d& landmarkEstimate,
    const GpuVec3d& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const Scalar errorThreshold,
    GpuVec3d& errors,
    GpuVec1i& outliers,
    GpuVec3d& Xcs,
    Scalar* chi);

Scalar computeActiveErrors_Line(
    const GpuVecSe3d& poseEstimate,
    const GpuVec<PointToLineMatch<double>>& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    GpuVec1d& errors,
    GpuVec3d& Xcs,
    Scalar* chi);

Scalar computeActiveErrors_Plane(
    const GpuVecSe3d& poseEstimate,
    const GpuVec<PointToPlaneMatch<double>>& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    GpuVec1d& errors,
    GpuVec3d& Xcs,
    Scalar* chi);

Scalar computeActiveErrors_PriorPose(
    const GpuVecSe3d& poseEstimate,
    const GpuVecSe3d& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    GpuVec6d& errors,
    GpuVec3d& Xcs,
    Scalar* chi);

void constructQuadraticForm_Line(
    const GpuVecSe3d& se3,
    GpuVec1d& errors,
    const GpuVec<PointToLineMatch<double>>& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const GpuVec1i& edge2Hpl,
    const GpuVec1b& flags,
    GpuPxPBlockVec& Hpp,
    GpuPx1BlockVec& bp,
    GpuLxLBlockVec& Hll,
    GpuLx1BlockVec& bl,
    GpuHplBlockMat& Hpl);

void constructQuadraticForm_Plane(
    const GpuVecSe3d& se3,
    GpuVec1d& errors,
    const GpuVec<PointToPlaneMatch<double>>& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const GpuVec1i& edge2Hpl,
    const GpuVec1b& flags,
    GpuPxPBlockVec& Hpp,
    GpuPx1BlockVec& bp,
    GpuLxLBlockVec& Hll,
    GpuLx1BlockVec& bl,
    GpuHplBlockMat& Hpl);

} // namespace gpu

} // namespace cugo

#endif // !__CUDA_BLOCK_SOLVER_H__
