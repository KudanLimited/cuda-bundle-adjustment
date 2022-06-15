
#ifndef __CUDA_BLOCK_SOLVER_IMPL_H__
#define __CUDA_BLOCK_SOLVER_IMPL_H__

#include "cuda_bundle_adjustment.h"
#include "cuda_linear_solver.h"
#include "device_buffer.h"
#include "device_matrix.h"
#include "sparse_block_matrix.h"

#include <array>
#include <vector>


namespace cugo
{
// forward declerations
class BaseEdge;
class BaseEdgeSet;

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
        PROF_ITEM_SOLVE_HPP,
        PROF_ITEM_NUM
    };

    void
    initialize(CameraParams* camera, const EdgeSetVec& edgeSets, const VertexSetVec& vertexSets);

    void buildStructure(const EdgeSetVec& edgeSets, const VertexSetVec& vertexSets);

    double computeErrors(const EdgeSetVec& edgeSets, const VertexSetVec& vertexSets);

    void buildSystem(const EdgeSetVec& edgeSets, const VertexSetVec& vertexSets);
    double maxDiagonal();

    void setLambda(double lambda);
    void restoreDiagonal();

    bool solve();
    void update(const VertexSetVec& vertexSets);

    double computeScale(double lambda);

    void push();
    void pop();

    void finalize(const VertexSetVec& vertexSets);
    void getTimeProfile(TimeProfile& prof) const;

    static inline uint8_t makeEdgeFlag(bool fixedP, bool fixedL)
    {
        uint8_t flag = 0;
        if (fixedP)
            flag |= EDGE_FLAG_FIXED_P;
        if (fixedL)
            flag |= EDGE_FLAG_FIXED_L;
        return flag;
    }

    int nedges() const { return nedges_; }

    ////////////////////////////////////////////////////////////////////////////////////
    // host buffers
    ////////////////////////////////////////////////////////////////////////////////////

private:
    bool doSchur;

    int nedges_;

    // graph components
    std::vector<BaseEdge*> baseEdges_;

    // block matrices
    HplSparseBlockMatrix Hpl_;
    HppSparseBlockMatrix Hpp_;
    HschurSparseBlockMatrix Hsc_;
    std::unique_ptr<LinearSolver> linearSolver_;
    std::vector<HplBlockPos> HplBlockPos_;

    ////////////////////////////////////////////////////////////////////////////////////
    // device buffers
    ////////////////////////////////////////////////////////////////////////////////////

    // solution vectors
    GpuVec1d d_solution_, d_solutionBackup_;

    // edge information
    GpuVec1i d_edge2Hpl_;

    // solution increments Δx = [Δxp Δxl]
    GpuVec1d d_x_;
    GpuPx1BlockVec d_xp_;
    GpuLx1BlockVec d_xl_;

    // coefficient matrix of linear system
    // | Hpp  Hpl ||Δxp| = |-bp|
    // | HplT Hll ||Δxl|   |-bl|
    GpuHppBlockMat d_Hpp_;
    GpuHllBlockMat d_Hll_;
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
    GpuVec1d d_HppCSR_;
    GpuVec1i d_BSR2CSR_;

    // temporary buffer
    DeviceBuffer<Scalar> d_chi_;
    GpuVec1i d_nnzPerCol_;

    ////////////////////////////////////////////////////////////////////////////////////
    // statistics
    ////////////////////////////////////////////////////////////////////////////////////

    std::vector<double> profItems_;
};

} // namespace cugo

#endif