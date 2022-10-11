#pragma once

#include "arena.h"
#include "cuda_graph_optimisation.h"
#include "cuda_linear_solver.h"
#include "device_buffer.h"
#include "device_matrix.h"
#include "fixed_vector.h"
#include "graph_optimisation_options.h"
#include "sparse_block_matrix.h"

#include <array>
#include <vector>

namespace cugo
{
// forward declerations
class BaseEdge;
class BaseEdgeSet;

/**
 * @brief The block solver for the optimisation problem.
 */
class BlockSolver
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

    static constexpr int HBLOCKPOS_ARENA_SIZE = 100000000;
    static constexpr int POSE_VERTEX_RESERVE_SIZE = 20000;
    static constexpr int LANDMARK_VERTEX_RESERVE_SIZE = 20000;

    BlockSolver() = delete;
    BlockSolver(GraphOptimisationOptions& options) : options(options), doSchur_(false), nedges_(0) {}

    /**
     * @brief Initialise the block solver. This will clear the old estimate values (if an
     * optimisation has already been carried out.)
     * @param edgeSets Edge sets associated with the graph optimisation.
     * @param vertexSets Vertex sets associated with the graph optimisation.
     */
    void initialize(
        const EdgeSetVec& edgeSets,
        const VertexSetVec& vertexSets,
        std::array<cudaStream_t, 3>& streams);

    /**
     * @brief BUilds the graph structure based on the vertices and connecting edges.
     * @param edgeSets Edge sets associated with the graph optimisation.
     * @param streams A vector of CUDA streams
     */
    void buildStructure(
        const EdgeSetVec& edgeSets,
        const VertexSetVec& vertexSets,
        std::array<cudaStream_t, 3>& streams);

    /**
     * @brief Compute the error of the graph.
     * @param edgeSets Edge sets associated with the graph optimisation.
     * @param vertexSets Vertex sets associated with the graph optimisation.
     * @param streams A vector of CUDA streams
     * @return double The calculated error value.
     */
    double computeErrors(
        const EdgeSetVec& edgeSets,
        const VertexSetVec& vertexSets,
        std::array<cudaStream_t, 3>& streams);

    /**
     * @brief Compute the quadratic equation of the graph.
     * @param edgeSets Edge sets associated with the graph optimisation.
     * @param vertexSets Vertex sets associated with the graph optimisation.
     * @param streams  A vector of CUDA streams
     */
    void buildSystem(
        const EdgeSetVec& edgeSets,
        const VertexSetVec& vertexSets,
        std::array<cudaStream_t, 3>& streams);

    /**
     * @brief Compute the maximum value on the hessain matrix as computed by @see buildSystem
     * @return The maximum value on the diagonal of the H matrix.
     */
    double maxDiagonal(std::array<cudaStream_t, 3>& streams);

    /**
     * @brief Set the lambda value on the H matrix diagonal.
     * @param lambda The lambda value to set.
     */
    void setLambda(double lambda, std::array<cudaStream_t, 3>& streams);

    /**
     * @brief Restore the H matrix diagonal back to the values before the @see setLambda call.
     *
     */
    void restoreDiagonal(std::array<cudaStream_t, 3>& streams);

    /**
     * @brief Solve the linear equation Ax = b
     * @return If the equation is successful, returns true.
     */
    bool solve(std::array<cudaStream_t, 3>& streams);

    void update(const VertexSetVec& vertexSets, std::array<cudaStream_t, 3>& streams);

    /**
    * @brief Remove outliers from the edgeSet if the outlier threshold is set.
    * @param edgeSets The edgesets to update.
    */
    void updateEdges(const EdgeSetVec& edgeSets);

    double computeScale(double lambda);

    /**
     * @brief Push the current H matrix into the backup buffer
     */
    void push();

    /**
     * @brief Retrieve the H matrix from the backup buffer and use as current matrix
     */
    void pop();

    /** 
     * @brief Clears all of the structures and variables set by the initialisation. 
    **/
    void clear() noexcept;

    /**
     * @brief Copy the new estimates into the host container.
     * @param vertexSets The vertex sets which will be updated.
     */
    void finalize(const VertexSetVec& vertexSets);

    /**
     * @brief Get the Time Profile object
     * @param prof
     */
    void getTimeProfile(TimeProfile& prof) const;

    /**
     * @brief The total number of edges that will be used by the block solver.
     * @return The total edge count.
     */
    int nedges() const { return nedges_; }

    /**
     * @brief Create a bit-flag that is used on the device to determine if pose or landmark vertices
     * are fixed.
     * @param fixedP States if the pose veretex is fixed.
     * @param fixedL States if the landmark veretex is fixed.
     * @return uint8_t The generated edge fixed bit flag
     */
    static inline uint8_t makeEdgeFlag(bool fixedP, bool fixedL)
    {
        uint8_t flag = 0;
        if (fixedP)
        {
            flag |= EDGE_FLAG_FIXED_P;
        }
        if (fixedL)
        {
            flag |= EDGE_FLAG_FIXED_L;
        }
        return flag;
    }

private:
    // a reference to the options used by the graph optimiser
    GraphOptimisationOptions& options;

    bool doSchur_;
    int nedges_;

    std::vector<BaseVertex*> verticesP_;
    std::vector<BaseVertex*> verticesL_;

    async_vector<HplBlockPos> HplblockPos_;

    /// active sizes only for pose and landmark
    size_t numP_ = 0;
    size_t numL_ = 0;

    /// block matrices
    HplSparseBlockMatrix Hpl_;
    HppSparseBlockMatrix Hpp_;
    HschurSparseBlockMatrix Hsc_;
    std::unique_ptr<LinearSolver> linearSolver_;

    // device buffers
    /// solution vectors
    GpuVec1d d_solution_, d_solutionBackup_;

    /// edge information
    GpuVec1i d_edge2Hpl_;

    /// solution increments Δx = [Δxp Δxl]
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

    /// conversion matrix storage format BSR to CSR
    GpuVec1d d_HscCSR_;
    GpuVec1d d_HppCSR_;
    GpuVec1i d_BSR2CSR_;

    /// temporary buffer
    DeviceBuffer<Scalar> d_chi_;
    GpuVec1i d_nnzPerCol_;

    std::vector<double> profItems_;
};

} // namespace cugo
