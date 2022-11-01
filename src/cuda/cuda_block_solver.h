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

#include "cuda/cuda_constants.h"
#include "device_matrix.h"
#include "fixed_vector.h"
#include "graph_optimisation_options.h"
#include "measurements.h"
#include "cuda_device.h"
#include "robust_kernel.h"

namespace cugo
{

namespace gpu
{

// needs moving to its own file
namespace detail
{

template <class F, class... Args>
F for_each_argument_address(F f, Args&&... args)
{
    (f((void*)&std::forward<Args>(args)), ...);
    return f;
}


} // namespace detail


template <int N>
using Vecxd = Vec<Scalar, N>;

template <int N>
using GpuVecxd = GpuVec<Vecxd<N>>;

// kernel functions
void waitForKernelCompletion();

void recordEvent(const CudaDeviceInfo& info);

void waitForEvent(const cudaEvent_t event);

void createRkFunction(
    RobustKernelType type, const GpuVec<Scalar>& d_delta, const CudaDeviceInfo& deviceInfo);

void buildHplStructure(
    GpuVec3i& blockpos,
    GpuHplBlockMat& Hpl,
    GpuVec1i& indexPL,
    GpuVec1i& nnzPerCol,
    const CudaDeviceInfo& deviceInfo1,
    const CudaDeviceInfo& deviceInfo2);

void findHschureMulBlockIndices(
    const GpuHplBlockMat& Hpl,
    const GpuHscBlockMat& Hsc,
    GpuVec3i& mulBlockIds,
    const CudaDeviceInfo& deviceInfo);

Scalar maxDiagonal(
    const GpuPxPBlockVec& Hpp, Scalar* d_maxD, Scalar* h_tmpMax, const CudaDeviceInfo& deviceInfo);
Scalar maxDiagonal(
    const GpuLxLBlockVec& Hll, Scalar* d_maxD, Scalar* h_tmpMax, const CudaDeviceInfo& deviceInfo);

void addLambda(
    GpuPxPBlockVec& Hpp, Scalar lambda, GpuPx1BlockVec& backup, const CudaDeviceInfo& deviceInfo);
void addLambda(
    GpuLxLBlockVec& Hll, Scalar lambda, GpuLx1BlockVec& backup, const CudaDeviceInfo& deviceInfo);

void restoreDiagonal(
    GpuPxPBlockVec& Hpp, const GpuPx1BlockVec& backup, const CudaDeviceInfo& deviceInfo);
void restoreDiagonal(
    GpuLxLBlockVec& Hll, const GpuLx1BlockVec& backup, const CudaDeviceInfo& deviceInfo);

void computeBschure(
    const GpuPx1BlockVec& bp,
    const GpuHplBlockMat& Hpl,
    const GpuLxLBlockVec& Hll,
    const GpuLx1BlockVec& bl,
    GpuPx1BlockVec& bsc,
    GpuLxLBlockVec& invHll,
    GpuPxLBlockVec& Hpl_invHll,
    const CudaDeviceInfo& deviceInfo);

void computeHschure(
    const GpuPxPBlockVec& Hpp,
    const GpuPxLBlockVec& Hpl_invHll,
    const GpuHplBlockMat& Hpl,
    const GpuVec3i& mulBlockIds,
    GpuHscBlockMat& Hsc,
    const CudaDeviceInfo& deviceInfo);

void convertHschureBSRToCSR(
    const GpuHscBlockMat& HscBSR,
    const GpuVec1i& BSR2CSR,
    GpuVec1d& HscCSR,
    const CudaDeviceInfo& deviceInfo);

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
    int* nnzPerRow,
    const CudaDeviceInfo& deviceInfo);

void permute(int size, const Scalar* src, Scalar* dst, const int* P);

void schurComplementPost(
    const GpuLxLBlockVec& invHll,
    const GpuLx1BlockVec& bl,
    const GpuHplBlockMat& Hpl,
    const GpuPx1BlockVec& xp,
    GpuLx1BlockVec& xl,
    const CudaDeviceInfo& deviceInfo);

void updatePoses(const GpuPx1BlockVec& xp, GpuVecSe3d& estimate, const CudaDeviceInfo& deviceInfo);

void updateLandmarks(const GpuLx1BlockVec& xl, GpuVec3d& Xws, const CudaDeviceInfo& deviceInfo);

void computeScale(
    const GpuVec1d& x,
    const GpuVec1d& b,
    Scalar* scale,
    Scalar lambda,
    const CudaDeviceInfo& deviceInfo);

void computeOutliers(
    int nedges,
    const Scalar errorTheshold,
    const Scalar* chiValues,
    int* outliers,
    const CudaDeviceInfo& deviceInfo);

template <int M>
void CUGO_API constructQuadraticForm_(
    const GpuVec3d& Xcs,
    const GpuVecSe3d& se3,
    GpuVecxd<M>& errors,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const GpuVec1i& edge2Hpl,
    const GpuVec1b& flags,
    const GpuVec5d& cameras,
    const RobustKernel& robustKernel,
    GpuPxPBlockVec& Hpp,
    GpuPx1BlockVec& bp,
    GpuLxLBlockVec& Hll,
    GpuLx1BlockVec& bl,
    GpuHplBlockMat& Hpl,
    const CudaDeviceInfo& deviceInfo);

template <int M>
Scalar CUGO_API computeActiveErrors_(
    const GpuVecSe3d& poseEstimate,
    const GpuVec3d& landmarkEstimate,
    const GpuVecxd<M>& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const GpuVec5d& cameras,
    const RobustKernel& robustKernel,
    GpuVecxd<M>& errors,
    GpuVec1i& outliers,
    GpuVec3d& Xcs,
    Scalar* chiValues,
    Scalar* chi,
    const CudaDeviceInfo& deviceInfo);

Scalar CUGO_API computeActiveErrors_DepthBa(
    const GpuVecSe3d& poseEstimate,
    const GpuVec3d& landmarkEstimate,
    const GpuVec3d& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    const GpuVec5d& cameras,
    const Scalar errorThreshold,
    const RobustKernel& robustKernel,
    GpuVec3d& errors,
    GpuVec1i& outliers,
    GpuVec3d& Xcs,
    Scalar* chi,
    const CudaDeviceInfo& deviceInfo);

Scalar CUGO_API computeActiveErrors_Line(
    const GpuVecSe3d& poseEstimate,
    const GpuVec<PointToLineMatch<double>>& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    GpuVec1d& errors,
    GpuVec3d& Xcs,
    Scalar* chi);

Scalar CUGO_API computeActiveErrors_Plane(
    const GpuVecSe3d& poseEstimate,
    const GpuVec<PointToPlaneMatch<double>>& measurements,
    const GpuVec1d& omegas,
    const GpuVec2i& edge2PL,
    GpuVec1d& errors,
    GpuVec3d& Xcs,
    Scalar* chi,
    const CudaDeviceInfo& deviceInfo);

void CUGO_API constructQuadraticForm_Line(
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

void CUGO_API constructQuadraticForm_Plane(
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
    GpuHplBlockMat& Hpl,
    const CudaDeviceInfo& deviceInfo);


} // namespace gpu

} // namespace cugo
