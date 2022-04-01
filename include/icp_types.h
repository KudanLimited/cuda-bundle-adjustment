#ifndef __ICP_TYPES_H__
#define __ICP_TYPES_H__

#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "optimisable_graph.h"
#include "measurements.h"
#include "cuda/cuda_block_solver.h"
#include "fixed_vector.h"

namespace cuba
{

class LineEdgeSet : public EdgeSet<1, PointToLineMatch<double>, PointToLineMatch<double>, PoseVertex>
{
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	LineEdgeSet() {}
	~LineEdgeSet() {}

    Scalar computeError(const VertexSetVec& vertexSets, Scalar* chi) override
    {
        GpuVecSe3d estimates = static_cast<PoseVertexSet*>(vertexSets[0])->getDeviceEstimates();
        return gpu::computeActiveErrors_Line(estimates, d_measurements, d_omegas, d_edge2PL, d_errors, d_Xcs, chi);
    }

    void constructQuadraticForm(const VertexSetVec& vertexSets,
		GpuPxPBlockVec& Hpp, GpuPx1BlockVec& bp, GpuLxLBlockVec& Hll, GpuLx1BlockVec& bl, GpuHplBlockMat& Hpl) override
    {
        PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSets[0]);    
        GpuVecSe3d se3_data = poseVertexSet->getDeviceEstimates();
		gpu::constructQuadraticForm_Line(se3_data, d_errors, d_measurements, d_omegas, d_edge2PL, d_edge2Hpl, d_edgeFlags, Hpp, bp, Hll, bl, Hpl);
    }

private:

};

class PlaneEdgeSet : public EdgeSet<1, PointToPlaneMatch<double>, PointToPlaneMatch<double>, PoseVertex>
{
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	PlaneEdgeSet() {}
	~PlaneEdgeSet() {}

    Scalar computeError(const VertexSetVec& vertexSets, Scalar* chi) override
    {
        GpuVecSe3d estimates = static_cast<PoseVertexSet*>(vertexSets[0])->getDeviceEstimates();
        return gpu::computeActiveErrors_Plane(estimates, d_measurements, d_omegas, d_edge2PL, d_errors, d_Xcs, chi);
    }

    void constructQuadraticForm(const VertexSetVec& vertexSets,
		GpuPxPBlockVec& Hpp, GpuPx1BlockVec& bp, GpuLxLBlockVec& Hll, GpuLx1BlockVec& bl, GpuHplBlockMat& Hpl) override
    {
        PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSets[0]);    
        GpuVecSe3d se3_data = poseVertexSet->getDeviceEstimates();
		gpu::constructQuadraticForm_Plane(se3_data, d_errors, d_measurements, d_omegas, d_edge2PL, d_edge2Hpl, d_edgeFlags, Hpp, bp, Hll, bl, Hpl);
    }

private:

};

class PlaneEdge : public Edge<1, cuba::PointToPlaneMatch<double>, cuba::PoseVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void* getMeasurement() override { return static_cast<void*>(&measurement); }
};


class LineEdge : public Edge<1, cuba::PointToLineMatch<double>, cuba::PoseVertex>
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
        return gpu::computeActiveErrors_PriorPose(poseEstimates, d_measurements, d_omegas, d_edge2PL, d_errors, d_Xcs, chi);
    }

    

private:

};*/



} // namespace cuba

#endif // !__ICP_TYPES_H__
