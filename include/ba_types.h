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

#ifndef __BA_TYPES_H__
#define __BA_TYPES_H__

#include "macro.h"
#include "optimisable_graph.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace cugo
{
class CUGO_API StereoEdgeSet : public EdgeSet<3, maths::Vec3d, Vec3d, PoseVertex, LandmarkVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    StereoEdgeSet() {}
    ~StereoEdgeSet() {}

    Scalar computeError(const VertexSetVec& vertexSets, Scalar* chi, cudaStream_t stream) override
    {
        GpuVecSe3d poseEstimateData;
        GpuVec3d landmarkEstimateData;

        PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSets[0]);
        poseEstimateData = poseVertexSet->getDeviceEstimates();
        LandmarkVertexSet* lmVertexSet = static_cast<LandmarkVertexSet*>(vertexSets[1]);
        landmarkEstimateData = lmVertexSet->getDeviceEstimates();

        return gpu::computeActiveErrors_<3>(
            poseEstimateData,
            landmarkEstimateData,
            d_measurements,
            d_omegas,
            d_edge2PL,
            outlierThreshold,
            d_errors,
            d_outliers,
            d_Xcs,
            chi,
            stream);
    }

    void constructQuadraticForm(
        const VertexSetVec& vertexSets,
        GpuPxPBlockVec& Hpp,
        GpuPx1BlockVec& bp,
        GpuLxLBlockVec& Hll,
        GpuLx1BlockVec& bl,
        GpuHplBlockMat& Hpl,
        cudaStream_t stream) override
    {
        // NOTE: This assumes the pose vertex is of the SE3 form - also would break if more than one
        // pose vertexset.
        assert(is_initialised == true);
        for (auto* vertexSet : vertexSets)
        {
            if (!vertexSet->isMarginilised())
            {
                // safe to be 100% sure this is a poseVertexSet?
                PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSet);
                GpuVecSe3d se3_data = poseVertexSet->getDeviceEstimates();
                gpu::constructQuadraticForm_<3>(
                    d_Xcs,
                    se3_data,
                    d_errors,
                    d_omegas,
                    d_edge2PL,
                    d_edge2Hpl,
                    d_edgeFlags,
                    Hpp,
                    bp,
                    Hll,
                    bl,
                    Hpl,
                    stream);
            }
        }
    }

private:
};

class CUGO_API MonoEdgeSet : public EdgeSet<2, maths::Vec2d, Vec2d, PoseVertex, LandmarkVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MonoEdgeSet() {}
    ~MonoEdgeSet() {}

    Scalar computeError(const VertexSetVec& vertexSets, Scalar* chi, cudaStream_t stream) override
    {
        GpuVecSe3d poseEstimateData;
        GpuVec3d landmarkEstimateData;

        PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSets[0]);
        poseEstimateData = poseVertexSet->getDeviceEstimates();
        LandmarkVertexSet* lmVertexSet = static_cast<LandmarkVertexSet*>(vertexSets[1]);
        landmarkEstimateData = lmVertexSet->getDeviceEstimates();

        return gpu::computeActiveErrors_<2>(
            poseEstimateData,
            landmarkEstimateData,
            d_measurements,
            d_omegas,
            d_edge2PL,
            outlierThreshold,
            d_errors,
            d_outliers,
            d_Xcs,
            chi,
            stream);
    }

    void constructQuadraticForm(
        const VertexSetVec& vertexSets,
        GpuPxPBlockVec& Hpp,
        GpuPx1BlockVec& bp,
        GpuLxLBlockVec& Hll,
        GpuLx1BlockVec& bl,
        GpuHplBlockMat& Hpl,
        cudaStream_t stream) override
    {
        // NOTE: This assumes the pose vertex is of the SE3 form - also would break if more than one
        // pose vertexset.
        assert(is_initialised == true);
        for (auto* vertexSet : vertexSets)
        {
            if (!vertexSet->isMarginilised())
            {
                // safe to be 100% sure this is a poseVertexSet?
                PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSet);
                GpuVecSe3d se3_data = poseVertexSet->getDeviceEstimates();
                gpu::constructQuadraticForm_<2>(
                    d_Xcs,
                    se3_data,
                    d_errors,
                    d_omegas,
                    d_edge2PL,
                    d_edge2Hpl,
                    d_edgeFlags,
                    Hpp,
                    bp,
                    Hll,
                    bl,
                    Hpl,
                    stream);
            }
        }
    }

private:
};

class CUGO_API DepthEdgeSet : public EdgeSet<3, maths::Vec3d, Vec3d, PoseVertex, LandmarkVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    DepthEdgeSet() {}
    ~DepthEdgeSet() {}

    Scalar computeError(const VertexSetVec& vertexSets, Scalar* chi, cudaStream_t stream) override
    {
        GpuVecSe3d poseEstimateData;
        GpuVec3d landmarkEstimateData;

        PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSets[0]);
        poseEstimateData = poseVertexSet->getDeviceEstimates();
        LandmarkVertexSet* lmVertexSet = static_cast<LandmarkVertexSet*>(vertexSets[1]);
        landmarkEstimateData = lmVertexSet->getDeviceEstimates();

        return gpu::computeActiveErrors_DepthBa(
            poseEstimateData,
            landmarkEstimateData,
            d_measurements,
            d_omegas,
            d_edge2PL,
            outlierThreshold,
            d_errors,
            d_outliers,
            d_Xcs,
            chi,
            stream);
    }

    void constructQuadraticForm(
        const VertexSetVec& vertexSets,
        GpuPxPBlockVec& Hpp,
        GpuPx1BlockVec& bp,
        GpuLxLBlockVec& Hll,
        GpuLx1BlockVec& bl,
        GpuHplBlockMat& Hpl,
        cudaStream_t stream) override
    {
        // NOTE: This assumes the pose vertex is of the SE3 form - also would break if more than one
        // pose vertexset.
        assert(is_initialised == true);
        for (auto* vertexSet : vertexSets)
        {
            if (!vertexSet->isMarginilised())
            {
                // safe to be 100% sure this is a poseVertexSet?
                PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSet);
                GpuVecSe3d se3_data = poseVertexSet->getDeviceEstimates();
                gpu::constructQuadraticForm_<3>(
                    d_Xcs,
                    se3_data,
                    d_errors,
                    d_omegas,
                    d_edge2PL,
                    d_edge2Hpl,
                    d_edgeFlags,
                    Hpp,
                    bp,
                    Hll,
                    bl,
                    Hpl,
                    stream);
            }
        }
    }
};

/** @brief Edge with 2-dimensional measurement (monocular observation).
 */
class CUGO_API MonoEdge : public Edge<2, maths::Vec2d, PoseVertex, LandmarkVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void* getMeasurement() override { return static_cast<void*>(measurement.data()); }
};

/** @brief Edge with 3-dimensional measurement (stereo observation).
 */
class CUGO_API StereoEdge : public Edge<3, maths::Vec3d, PoseVertex, LandmarkVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void* getMeasurement() override { return static_cast<void*>(measurement.data()); }
};

/** @brief Edge with 3-dimensional measurement (depth observation).
 */
class CUGO_API DepthEdge : public Edge<3, maths::Vec3d, PoseVertex, LandmarkVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void* getMeasurement() override { return static_cast<void*>(measurement.data()); }
};

} // namespace cugo

#endif // !__BA_TYPES_H__
