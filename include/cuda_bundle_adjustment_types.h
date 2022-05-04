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

#ifndef __CUDA_BUNDLE_ADJUSTMENT_TYPES_H__
#define __CUDA_BUNDLE_ADJUSTMENT_TYPES_H__

#include "optimisable_graph.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace cuba
{
class StereoEdgeSet : public EdgeSet<3, maths::Vec3d, Vec3d, PoseVertex, LandmarkVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    StereoEdgeSet() {}
    ~StereoEdgeSet() {}

    Scalar computeError(const VertexSetVec& vertexSets, Scalar* chi) override
    {
        assert(is_initialised == true);
        assert(vertexSets.size() == 2);
        GpuVecSe3d poseEstimateData;
        GpuVec3d landmarkEstimateData;
        for (auto* vertexSet : vertexSets)
        {
            if (!vertexSet->isMarginilised())
            {
                PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSet);
                poseEstimateData = poseVertexSet->getDeviceEstimates();
            }
            else
            {
                LandmarkVertexSet* lmVertexSet = static_cast<LandmarkVertexSet*>(vertexSet);
                landmarkEstimateData = lmVertexSet->getDeviceEstimates();
            }
        }
        return gpu::computeActiveErrors_<3>(
            poseEstimateData,
            landmarkEstimateData,
            d_measurements,
            d_omegas,
            d_edge2PL,
            d_errors,
            d_Xcs,
            chi);
    }

    void constructQuadraticForm(
        const VertexSetVec& vertexSets,
        GpuHppBlockMat& Hpp,
        GpuPx1BlockVec& bp,
        GpuHllBlockMat& Hll,
        GpuLx1BlockVec& bl,
        GpuHplBlockMat& Hpl) override
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
                    Hpl);
            }
        }
    }

private:
};


class MonoEdgeSet : public EdgeSet<2, maths::Vec2d, Vec2d, PoseVertex, LandmarkVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MonoEdgeSet() {}
    ~MonoEdgeSet() {}

    Scalar computeError(const VertexSetVec& vertexSets, Scalar* chi) override
    {
        assert(is_initialised == true);
        assert(vertexSets.size() == 2);
        GpuVecSe3d poseEstimateData;
        GpuVec3d landmarkEstimateData;
        for (auto* vertexSet : vertexSets)
        {
            if (!vertexSet->isMarginilised())
            {
                PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSet);
                poseEstimateData = poseVertexSet->getDeviceEstimates();
            }
            else
            {
                LandmarkVertexSet* lmVertexSet = static_cast<LandmarkVertexSet*>(vertexSet);
                landmarkEstimateData = lmVertexSet->getDeviceEstimates();
            }
        }
        return gpu::computeActiveErrors_<2>(
            poseEstimateData,
            landmarkEstimateData,
            d_measurements,
            d_omegas,
            d_edge2PL,
            d_errors,
            d_Xcs,
            chi);
    }

    void constructQuadraticForm(
        const VertexSetVec& vertexSets,
        GpuHppBlockMat& Hpp,
        GpuPx1BlockVec& bp,
        GpuHllBlockMat& Hll,
        GpuLx1BlockVec& bl,
        GpuHplBlockMat& Hpl) override
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
                    Hpl);
            }
        }
    }

private:
};

/** @brief Edge with 2-dimensional measurement (monocular observation).
 */
class MonoEdge : public Edge<2, maths::Vec2d, PoseVertex, LandmarkVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void* getMeasurement() override { return static_cast<void*>(measurement.data()); }
};

/** @brief Edge with 3-dimensional measurement (stereo observation).
 */
class StereoEdge : public Edge<3, maths::Vec3d, PoseVertex, LandmarkVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void* getMeasurement() override { return static_cast<void*>(measurement.data()); }
};

} // namespace cuba

#endif // !__CUDA_BUNDLE_ADJUSTMENT_TYPES_H__
