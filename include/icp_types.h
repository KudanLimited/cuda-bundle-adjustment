#ifndef __ICP_TYPES_H__
#define __ICP_TYPES_H__

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
    : public EdgeSet<1, PointToLineMatch<double>, PointToLineMatch<double>, PoseVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LineEdgeSet() {}
    ~LineEdgeSet() {}

    Scalar computeError(const VertexSetVec& vertexSets, Scalar* chi) override
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
        GpuHplBlockMat& Hpl) override
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

private:
};


class CUGO_API PlaneEdgeSet
    : public EdgeSet<1, PointToPlaneMatch<double>, PointToPlaneMatch<double>, PoseVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PlaneEdgeSet() {}
    ~PlaneEdgeSet() {}

    Scalar computeError(const VertexSetVec& vertexSets, Scalar* chi) override
    {
        GpuVecSe3d estimates = static_cast<PoseVertexSet*>(vertexSets[0])->getDeviceEstimates();
        return gpu::computeActiveErrors_Plane(
            estimates, d_measurements, d_omegas, d_edge2PL, d_errors, d_Xcs, chi);
    }

    void constructQuadraticForm(
        const VertexSetVec& vertexSets,
        GpuPxPBlockVec& Hpp,
        GpuPx1BlockVec& bp,
        GpuLxLBlockVec& Hll,
        GpuLx1BlockVec& bl,
        GpuHplBlockMat& Hpl) override
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
            Hpl);
    }

private:
};


class CUGO_API PlaneEdge : public Edge<1, cugo::PointToPlaneMatch<double>, cugo::PoseVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void* getMeasurement() override { return static_cast<void*>(&measurement); }
};


class CUGO_API LineEdge : public Edge<1, cugo::PointToLineMatch<double>, cugo::PoseVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void* getMeasurement() override { return static_cast<void*>(&measurement); }
};

/*
/
class PriorPoseEdge : public EdgeSet<6, maths::Se3D, Se3D, PoseVertex>
{
public:

    PriorPoseEdge() = default;

    void computeError(const VertexSetVec& vertexSets, Scalar* chi) override
    {
        GpuVecSe3d estimates = static_cast<PoseVertexSet*>(vertexSets[0])->getEstimates();
        return gpu::computeActiveErrors_PriorPose(poseEstimates, d_measurements, d_omegas,
d_edge2PL, d_errors, d_Xcs, chi);
    }



private:

};*/


} // namespace cugo

#endif // !__ICP_TYPES_H__
