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

#include "macro.h"
#include "optimisable_graph.h"
#include "cuda_device.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace cugo
{

class CUGO_API StereoEdgeSet : public EdgeSet<3, Vec3d, PoseVertex, LandmarkVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    StereoEdgeSet() {}
    ~StereoEdgeSet() {}

    Scalar computeError(
        const VertexSetVec& vertexSets,
        Scalar* chi, const CudaDeviceInfo& deviceInfo) override
    {
        PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSets[0]);
        GpuVecSe3d poseEstimateData = poseVertexSet->getDeviceEstimates();
        LandmarkVertexSet* lmVertexSet = static_cast<LandmarkVertexSet*>(vertexSets[1]);
        GpuVec3d landmarkEstimateData = lmVertexSet->getDeviceEstimates();

        return gpu::computeActiveErrors_<3>(
            poseEstimateData,
            landmarkEstimateData,
            d_measurements,
            d_omegas,
            d_edge2PL,
            d_cameras,
            kernel,
            d_outliers,
            d_errors,
            d_Xcs,
            d_chiValues,
            chi,
            deviceInfo);
    }

    void constructQuadraticForm(
        const VertexSetVec& vertexSets,
        GpuPxPBlockVec& Hpp,
        GpuPx1BlockVec& bp,
        GpuLxLBlockVec& Hll,
        GpuLx1BlockVec& bl,
        GpuHplBlockMat& Hpl,
        const CudaDeviceInfo& deviceInfo) override
    {
        // NOTE: This assumes the pose vertex is of the SE3 form - also would break if more than one
        // pose vertexset.
        // safe to be 100% sure this is a poseVertexSet?
        PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSets[0]);
        GpuVecSe3d se3_data = poseVertexSet->getDeviceEstimates();
        gpu::constructQuadraticForm_<3>(
            d_Xcs,
            se3_data,
            d_errors,
            d_omegas,
            d_edge2PL,
            d_edge2Hpl,
            d_edgeFlags,
            d_cameras,
            kernel,
            d_outliers,
            Hpp,
            bp,
            Hll,
            bl,
            Hpl,
            deviceInfo);
    }

private:
};

class CUGO_API MonoEdgeSet : public EdgeSet<2, Vec2d, PoseVertex, LandmarkVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MonoEdgeSet() {}
    ~MonoEdgeSet() {}

    Scalar computeError(
        const VertexSetVec& vertexSets,
        Scalar* chi, const CudaDeviceInfo& deviceInfo) override
    {
        PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSets[0]);
        GpuVecSe3d poseEstimateData = poseVertexSet->getDeviceEstimates();
        LandmarkVertexSet* lmVertexSet = static_cast<LandmarkVertexSet*>(vertexSets[1]);
        GpuVec3d landmarkEstimateData = lmVertexSet->getDeviceEstimates();

        return gpu::computeActiveErrors_<2>(
            poseEstimateData,
            landmarkEstimateData,
            d_measurements,
            d_omegas,
            d_edge2PL,
            d_cameras,
            kernel,
            d_outliers,
            d_errors,
            d_Xcs,
            d_chiValues,
            chi,
            deviceInfo);
    }

    void constructQuadraticForm(
        const VertexSetVec& vertexSets,
        GpuPxPBlockVec& Hpp,
        GpuPx1BlockVec& bp,
        GpuLxLBlockVec& Hll,
        GpuLx1BlockVec& bl,
        GpuHplBlockMat& Hpl,
        const CudaDeviceInfo& deviceInfo) override
    {
        // NOTE: This assumes the pose vertex is of the SE3 form and is in index position zero
        PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSets[0]);
        GpuVecSe3d se3_data = poseVertexSet->getDeviceEstimates();
        gpu::constructQuadraticForm_<2>(
            d_Xcs,
            se3_data,
            d_errors,
            d_omegas,
            d_edge2PL,
            d_edge2Hpl,
            d_edgeFlags,
            d_cameras,
            kernel,
            d_outliers,
            Hpp,
            bp,
            Hll,
            bl,
            Hpl,
            deviceInfo);

    }

private:
};

class CUGO_API DepthEdgeSet : public EdgeSet<3, Vec3d, PoseVertex, LandmarkVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    DepthEdgeSet() {}
    ~DepthEdgeSet() {}

    Scalar computeError(
        const VertexSetVec& vertexSets,
        Scalar* chi, const CudaDeviceInfo& deviceInfo) override
    {
        PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSets[0]);
        GpuVecSe3d poseEstimateData = poseVertexSet->getDeviceEstimates();
        LandmarkVertexSet* lmVertexSet = static_cast<LandmarkVertexSet*>(vertexSets[1]);
        GpuVec3d landmarkEstimateData = lmVertexSet->getDeviceEstimates();

        return gpu::computeActiveErrors_DepthBa(
            poseEstimateData,
            landmarkEstimateData,
            d_measurements,
            d_omegas,
            d_edge2PL,
            d_cameras,
            outlierThreshold,
            kernel,
            d_errors,
            d_outliers,
            d_Xcs,
            chi,
            deviceInfo);
    }

    void constructQuadraticForm(
        const VertexSetVec& vertexSets,
        GpuPxPBlockVec& Hpp,
        GpuPx1BlockVec& bp,
        GpuLxLBlockVec& Hll,
        GpuLx1BlockVec& bl,
        GpuHplBlockMat& Hpl,
        const CudaDeviceInfo& deviceInfo) override
    {
        // NOTE: This assumes the pose vertex is of the SE3 form and is in index position zero
        PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSets[0]);
        GpuVecSe3d se3_data = poseVertexSet->getDeviceEstimates();
        gpu::constructQuadraticForm_<3>(
            d_Xcs,
            se3_data,
            d_errors,
            d_omegas,
            d_edge2PL,
            d_edge2Hpl,
            d_edgeFlags,
            d_cameras,
            kernel,
            d_outliers,
            Hpp,
            bp,
            Hll,
            bl,
            Hpl,
            deviceInfo);
    }
};

/** @brief Edge with 2-dimensional measurement (monocular observation).
 */
class CUGO_API MonoEdge : public Edge<2, Vec2d, PoseVertex, LandmarkVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void* getMeasurement() noexcept override { return static_cast<void*>(measurement.data); }
};

/** @brief Edge with 3-dimensional measurement (stereo observation).
 */
class CUGO_API StereoEdge : public Edge<3, Vec3d, PoseVertex, LandmarkVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void* getMeasurement() noexcept override { return static_cast<void*>(measurement.data); }
};

/** @brief Edge with 3-dimensional measurement (depth observation).
 */
class CUGO_API DepthEdge : public Edge<3, Vec3d, PoseVertex, LandmarkVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void* getMeasurement() noexcept override { return static_cast<void*>(measurement.data); }
};

} // namespace cugo
