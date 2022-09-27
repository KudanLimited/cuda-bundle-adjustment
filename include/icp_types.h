#pragma once

#include "cuda/cuda_block_solver.h"
#include "fixed_vector.h"
#include "measurements.h"
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
class CUGO_API LineEdgeSet
    : public EdgeSet<1, PointToLineMatch<double>, PoseVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LineEdgeSet() {}
    ~LineEdgeSet() {}

    Scalar computeError(const VertexSetVec& vertexSets, Scalar* chi, cudaStream_t stream) override
    {
        GpuVecSe3d estimates = static_cast<PoseVertexSet*>(vertexSets[0])->getDeviceEstimates();
        return gpu::computeActiveErrors_Line(
            estimates, d_measurements, d_omegas, d_edge2PL, d_errors, d_Xcs, chi);
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
        PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSets[0]);
        GpuVecSe3d se3_data = poseVertexSet->getDeviceEstimates();
        gpu::constructQuadraticForm_Line(
            se3_data,
            d_errors,
            d_measurements,
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
};


class CUGO_API PlaneEdgeSet
    : public EdgeSet<1, PointToPlaneMatch<double>, PoseVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PlaneEdgeSet() {}
    ~PlaneEdgeSet() {}

    Scalar computeError(const VertexSetVec& vertexSets, Scalar* chi, cudaStream_t stream) override
    {
        GpuVecSe3d estimates = static_cast<PoseVertexSet*>(vertexSets[0])->getDeviceEstimates();
        return gpu::computeActiveErrors_Plane(
            estimates, d_measurements, d_omegas, d_edge2PL, d_errors, d_Xcs, chi, stream);
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
        PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSets[0]);
        GpuVecSe3d se3_data = poseVertexSet->getDeviceEstimates();
        gpu::constructQuadraticForm_Plane(
            se3_data,
            d_errors,
            d_measurements,
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
};


class CUGO_API PlaneEdge : public Edge<1, cugo::PointToPlaneMatch<double>, cugo::PoseVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void* getMeasurement() noexcept override { return static_cast<void*>(&measurement); }
};


class CUGO_API LineEdge : public Edge<1, cugo::PointToLineMatch<double>, cugo::PoseVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void* getMeasurement() noexcept override { return static_cast<void*>(&measurement); }
};

} // namespace cugo
