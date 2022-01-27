
#ifndef __CUDA_BLOCK_SOLVER_IMPL_H__
#define __CUDA_BLOCK_SOLVER_IMPL_H__

#include <vector>
#include <array>

#include "cuda_bundle_adjustment.h"
#include "device_buffer.h"
#include "device_matrix.h"
#include "sparse_block_matrix.h"
#include "cuda_linear_solver.h"


namespace cuba
{

// forward declerations
class BaseEdge;
class BaseEdgeSet;
struct PoseVertex;
struct LandmarkVertex;

/** @brief Implementation of Block solver.
*/
class CudaBlockSolver
{
public:

	enum ProfileItem
	{
		PROF_ITEM_INITIALIZE,
		PROF_ITEM_BUILD_STRUCTURE,
		PROF_ITEM_COMPUTE_ERROR,
		PROF_ITEM_BUILD_SYSTEM,
		PROF_ITEM_SCHUR_COMPLEMENT,
		PROF_ITEM_DECOMP_SYMBOLIC,
		PROF_ITEM_DECOMP_NUMERICAL,
		PROF_ITEM_UPDATE,
		PROF_ITEM_NUM
	};

	struct PLIndex
	{
		int P, L;
		PLIndex(int P = 0, int L = 0) : P(P), L(L) {}
	};

	void clear();

	void initialize(std::array<BaseEdgeSet*, 6>& edgeSets, std::map<int, PoseVertex*>& vertexMapP, 
		std::map<int, LandmarkVertex*>& vertexMapL, CameraParams* camera);

	void buildStructure(const std::array<BaseEdgeSet*, 6>& edgeSets);

	double computeErrors(const std::array<BaseEdgeSet*, 6>& edgeSets);

	void buildSystem(const std::array<BaseEdgeSet*, 6>& edgeSets);
	double maxDiagonal();

	void setLambda(double lambda);
	void restoreDiagonal();

	bool solve();
	void update();

	double computeScale(double lambda);

	void push();
	void pop();

	void finalize();
	void getTimeProfile(TimeProfile& prof) const;

private:

	static inline uint8_t makeEdgeFlag(bool fixedP, bool fixedL);

	////////////////////////////////////////////////////////////////////////////////////
	// host buffers
	////////////////////////////////////////////////////////////////////////////////////

	// graph components
	std::vector<PoseVertex*> verticesP_;
	std::vector<LandmarkVertex*> verticesL_;
	std::vector<BaseEdge*> baseEdges_;
	int numP_, numL_, nedges_;

	// solution vectors
	std::vector<Vec4d> qs_;
	std::vector<Vec3d> ts_;
	std::vector<Vec3d> Xws_;

	// edge information
	std::vector<Scalar> omegas_;
	std::vector<PLIndex> edge2PL_;
	std::vector<uint8_t> edgeFlags_;

	// block matrices
	HplSparseBlockMatrix Hpl_;
	HschurSparseBlockMatrix Hsc_;
	SparseLinearSolver::Ptr linearSolver_;
	std::vector<HplBlockPos> HplBlockPos_;
	int nHplBlocks_;

	////////////////////////////////////////////////////////////////////////////////////
	// device buffers
	////////////////////////////////////////////////////////////////////////////////////

	// solution vectors
	GpuVec1d d_solution_, d_solutionBackup_;
	GpuVec4d d_qs_;
	GpuVec3d d_ts_, d_Xws_;

	// edge information
	GpuVec1i d_edge2Hpl_;

	// solution increments Δx = [Δxp Δxl]
	GpuVec1d d_x_;
	GpuPx1BlockVec d_xp_;
	GpuLx1BlockVec d_xl_;

	// coefficient matrix of linear system
	// | Hpp  Hpl ||Δxp| = |-bp|
	// | HplT Hll ||Δxl|   |-bl|
	GpuPxPBlockVec d_Hpp_;
	GpuLxLBlockVec d_Hll_;
	GpuHplBlockMat d_Hpl_;
	GpuVec3i d_HplBlockPos_;
	GpuVec1d d_b_;
	GpuPx1BlockVec d_bp_;
	GpuLx1BlockVec d_bl_;
	GpuPx1BlockVec d_HppBackup_;
	GpuLx1BlockVec d_HllBackup_;

	// schur complement of the H matrix
	// HSc = Hpp - Hpl*inv(Hll)*HplT
	// bSc = -bp + Hpl*inv(Hll)*bl
	GpuHscBlockMat d_Hsc_;
	GpuPx1BlockVec d_bsc_;
	GpuLxLBlockVec d_invHll_;
	GpuPxLBlockVec d_Hpl_invHll_;
	GpuVec3i d_HscMulBlockIds_;

	// conversion matrix storage format BSR to CSR
	GpuVec1d d_HscCSR_;
	GpuVec1i d_BSR2CSR_;

	// temporary buffer
	DeviceBuffer<Scalar> d_chi_;
	GpuVec1i d_nnzPerCol_;

	////////////////////////////////////////////////////////////////////////////////////
	// statistics
	////////////////////////////////////////////////////////////////////////////////////

	std::vector<double> profItems_;
};

}

#endif