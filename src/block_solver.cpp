#include "block_solver.h"

#include "cuda_linear_solver.h"
#include "optimisable_graph.h"
#include "profile.h"

#include <cstdint>

namespace cugo
{

void BlockSolver::initialize(
    const EdgeSetVec& edgeSets,
    const VertexSetVec& vertexSets)
{
    // ensure we have data to work with
    assert(edgeSets.size() > 0);
    assert(vertexSets.size() > 0);

    cudaDevice_.startTimingEvent();

    // prepare the solver for a fresh initialisation
    clear();

    verticesP_.reserve(POSE_VERTEX_RESERVE_SIZE);
    verticesL_.reserve(LANDMARK_VERTEX_RESERVE_SIZE);

    for (BaseVertexSet* vertexSet : vertexSets)
    {
        vertexSet->clearEstimates();

        if (!vertexSet->isMarginilised())
        {
            PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSet);
            poseVertexSet->generateEstimateData();
            auto& setData = poseVertexSet->get();
            verticesP_.insert(verticesP_.end(), setData.begin(), setData.end());
            numP_ += vertexSet->getActiveSize();
        }
        else
        {
            LandmarkVertexSet* lmVertexSet = static_cast<LandmarkVertexSet*>(vertexSet);
            lmVertexSet->generateEstimateData();
            auto& setData = lmVertexSet->get();
            verticesL_.insert(verticesL_.end(), setData.begin(), setData.end());
            numL_ += vertexSet->getActiveSize();
        }
    }
    
    // upload the estimates to the device
    int offset = 0;

    d_solution_.resize(verticesL_.size() * 3 + verticesP_.size() * 7);
    d_solutionBackup_.resize(d_solution_.size());

    for (BaseVertexSet* vertexSet : vertexSets)
    {
        if (!vertexSet->isMarginilised())
        {
            PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSet);
            poseVertexSet->mapEstimateData(d_solution_.data() + offset, cudaDevice_.getStreamAndEvent(1));
            offset += poseVertexSet->getDeviceEstimateSize() * 7;
        }
        else
        {
            LandmarkVertexSet* lmVertexSet = static_cast<LandmarkVertexSet*>(vertexSet);
            lmVertexSet->mapEstimateData(
                d_solution_.data() + offset, cudaDevice_.getStreamAndEvent(2));
            offset += lmVertexSet->getDeviceEstimateSize() * 3;
        }
    }

    // only perform schur if we have landmark vertices
    doSchur_ = (verticesL_.size() > 0) ? true : false;
    
    // setup the edge set estimation data, indices and flags on the host
    // Note: this is dependent on the vertex data so this must be initialised first
    int edgeIdOffset = 0;

    if (doSchur_)
    {
        HplblockPos_.reserve(HBLOCKPOS_ARENA_SIZE);
    }

    for (BaseEdgeSet* edgeSet : edgeSets)
    {
        if (!edgeSet->nedges())
        {
            continue;
        }

        // clear the edges and vertices from the last run.
        // Note: these calls do no memory de-allocating and do not
        // destroy or touch device buffers.
        edgeSet->clearDevice();

        edgeSet->init(HplblockPos_, edgeIdOffset, 0, doSchur_, options);
        nedges_ += edgeSet->nActiveEdges();
        edgeIdOffset += nedges_;
    }
    
    if (doSchur_)
    {
        d_edge2Hpl_.resize(nedges_);
    }

    int prevEdgeSize = 0;

    for (BaseEdgeSet* edgeSet : edgeSets)
    {
        if (!edgeSet->nedges())
        {
            continue;
        }
        if (doSchur_)
        {
            // data layout is pose data first followed by landmark
            int* edge2HplPtr = d_edge2Hpl_.data() + prevEdgeSize;
            edgeSet->mapDevice(options, cudaDevice_.getStreamAndEvent(0), edge2HplPtr);
            prevEdgeSize += edgeSet->nActiveEdges();
        }
        else
        {
            edgeSet->mapDevice(options, cudaDevice_.getStreamAndEvent(0));
        }
    }

    // initialise the linear solver
    if (!linearSolver_)
    {
        if (doSchur_)
        {
            linearSolver_ = std::make_unique<HscSparseLinearSolver>();
        }
        else
        {
            linearSolver_ = std::make_unique<DenseLinearSolver>();
        }
    }

    profItems_[PROF_ITEM_INITIALIZE] = cudaDevice_.stopTimingEvent();
}

void BlockSolver::buildStructure(
    const EdgeSetVec& edgeSets, 
    const VertexSetVec& vertexSets)
{
    cudaDevice_.startTimingEvent();

    // allocate device buffers
    d_x_.resize(numP_ * PDIM + numL_ * LDIM);
    d_b_.resize(numP_ * PDIM + numL_ * LDIM);

    if (doSchur_)
    {
        size_t nVertexBlockPos = HplblockPos_.size();

        d_Hpl_.resize(numP_, numL_);
        d_Hpl_.resizeNonZeros(nVertexBlockPos);
        d_nnzPerCol_.resize(numL_ + 1);
   
        // build Hpl block matrix structure
        d_HplBlockPos_.assignAsync(nVertexBlockPos, HplblockPos_.data(), cudaDevice_.getStream(3));
        gpu::buildHplStructure(
            d_HplBlockPos_,
            d_Hpl_,
            d_edge2Hpl_,
            d_nnzPerCol_,
            cudaDevice_.getStreamAndEvent(3),
            cudaDevice_.getStreamAndEvent(4));

        // build host Hschur sparse block matrix structure
        Hsc_.resize(numP_, numP_);
        Hsc_.constructFromVertices(verticesL_);
        Hsc_.convertBSRToCSR();

        // initialise the device schur hessian matrix
        d_Hsc_.resize(numP_, numP_);
        d_Hsc_.resizeNonZeros(Hsc_.nblocks());
        // TODO: use async upload - need to converted Eigen::VectorXi to a pinned memory address
        // version
        d_Hsc_.uploadAsync(nullptr, Hsc_.outerIndices(), Hsc_.innerIndices());

        // initialise the device landmark Hessian matrix - 
        // this is filled by the computation of the quadratic form
        d_Hll_.resize(numL_);
       
        d_HscCSR_.resize(Hsc_.nnzSymm());
        d_BSR2CSR_.assignAsync(Hsc_.nnzSymm(), (int*)Hsc_.BSR2CSR());

        d_HscMulBlockIds_.resize(Hsc_.nmulBlocks());
        gpu::findHschureMulBlockIndices(
            d_Hpl_, d_Hsc_, d_HscMulBlockIds_, cudaDevice_.getStreamAndEvent(3));

        d_bsc_.resize(numP_);
        d_Hpl_invHll_.resize(nVertexBlockPos);
        d_HllBackup_.resize(numL_);
        d_invHll_.resize(numL_);
        d_tmpMax_hll_.resize(40);

        d_xl_.map(numL_, d_x_.data() + numP_ * PDIM);
        d_bl_.map(numL_, d_b_.data() + numP_ * PDIM);
    }
    else
    {
        Hpp_.resize(numP_, numP_);
        Hpp_.constructFromVertices(verticesP_);
        Hpp_.convertBSRToCSR();
    }

    // initialise the device pose Hessian matrix -
    // this is filled by the computation of the quadratic form
    d_Hpp_.resize(numP_);

    d_xp_.map(numP_, d_x_.data());
    d_bp_.map(numP_, d_b_.data());

    d_HppBackup_.resize(numP_);
    d_chi_.resize(1);
    d_tmpMax_hpp_.resize(40);
   
    if (doSchur_)
    {
        HscSparseLinearSolver* sparseLinearSolver =
            static_cast<HscSparseLinearSolver*>(linearSolver_.get());

        // analyze pattern of Hschur matrix (symbolic decomposition)
        sparseLinearSolver->initialize(Hsc_, cudaDevice_.getStreamAndEvent(2));
    }
    else
    {
        DenseLinearSolver* denseLinearSolver = static_cast<DenseLinearSolver*>(linearSolver_.get());
        denseLinearSolver->initialize(Hpp_, cudaDevice_.getStream(2));
    }

    profItems_[PROF_ITEM_BUILD_STRUCTURE] = cudaDevice_.stopTimingEvent();
}

double BlockSolver::computeErrors(
    const EdgeSetVec& edgeSets,
    const VertexSetVec& vertexSets)
{
    cudaDevice_.startTimingEvent();

    Scalar accumChi = 0;
    uint8_t streamId = 0;
    for (BaseEdgeSet* edgeSet : edgeSets)
    {
        if (!edgeSet->nedges())
        {
            continue;
        }
        const Scalar chi = edgeSet->computeError(vertexSets, d_chi_, cudaDevice_.getStreamAndEvent(streamId++));
        assert(streamId < CudaDevice::MaxStreamCount);
        accumChi += chi;
    }

    profItems_[PROF_ITEM_COMPUTE_ERROR] = cudaDevice_.stopTimingEvent();
    
    return accumChi;
}

void BlockSolver::buildSystem(
    const EdgeSetVec& edgeSets,
    const VertexSetVec& vertexSets)
{
    // Build linear system about solution increments Δx
    // H*Δx = -b
    //
    // coefficient matrix are divided into blocks, and each block is calculated
    // | Hpp  Hpl ||Δxp| = |-bp|
    // | HplT Hll ||Δxl|   |-bl|

    cudaDevice_.startTimingEvent();

    d_Hpp_.fillZero();
    d_bp_.fillZero();

    if (doSchur_)
    {
        d_Hll_.fillZero();
        d_bl_.fillZero();
    }

    for (auto* edgeSet : edgeSets)
    {
        if (!edgeSet->nedges())
        {
            continue;
        }
        edgeSet->constructQuadraticForm(
            vertexSets, d_Hpp_, d_bp_, d_Hll_, d_bl_, d_Hpl_, cudaDevice_.getStreamAndEvent(0));
    }

    profItems_[PROF_ITEM_BUILD_SYSTEM] = cudaDevice_.stopTimingEvent();
}

double BlockSolver::maxDiagonal()
{
    const Scalar maxP = gpu::maxDiagonal(
        d_Hpp_, d_tmpMax_hpp_.data(), h_tmpMax_hpp_.values(), cudaDevice_.getStreamAndEvent(0));
    if (doSchur_)
    {
        const Scalar maxL = gpu::maxDiagonal(
            d_Hll_, d_tmpMax_hll_.data(), h_tmpMax_hll_.values(), cudaDevice_.getStreamAndEvent(1));
        return std::max(maxP, maxL);
    }
    return maxP;
}

void BlockSolver::setLambda(double lambda)
{
    gpu::addLambda(d_Hpp_, ScalarCast(lambda), d_HppBackup_, cudaDevice_.getStreamAndEvent(0));
    if (doSchur_)
    {
        gpu::addLambda(d_Hll_, ScalarCast(lambda), d_HllBackup_, cudaDevice_.getStreamAndEvent(1));
    }
}

void BlockSolver::restoreDiagonal()
{
    gpu::restoreDiagonal(d_Hpp_, d_HppBackup_, cudaDevice_.getStreamAndEvent(0));
    if (doSchur_)
    {
        gpu::restoreDiagonal(d_Hll_, d_HllBackup_, cudaDevice_.getStreamAndEvent(1));
    }
}

bool BlockSolver::solve()
{
    cudaDevice_.startTimingEvent();

    if (!doSchur_)
    {
        const bool success = linearSolver_->solve(d_Hpp_.values(), d_bp_.values(), d_xp_.values());
        if (!success)
        {
            return false;
        }

        profItems_[PROF_ITEM_SOLVE_HPP] = cudaDevice_.stopTimingEvent();
        return true;
    }

    // Schur complement
    // bSc = -bp + Hpl*Hll^-1*bl
    // HSc = Hpp - Hpl*Hll^-1*HplT
    gpu::computeBschure(
        d_bp_,
        d_Hpl_,
        d_Hll_,
        d_bl_,
        d_bsc_,
        d_invHll_,
        d_Hpl_invHll_,
        cudaDevice_.getStreamAndEvent(3));
    gpu::computeHschure(
        d_Hpp_, d_Hpl_invHll_, d_Hpl_, d_HscMulBlockIds_, d_Hsc_, cudaDevice_.getStreamAndEvent(3));

    // Solve linear equation about Δxp
    // HSc*Δxp = bp
    gpu::convertHschureBSRToCSR(d_Hsc_, d_BSR2CSR_, d_HscCSR_, cudaDevice_.getStreamAndEvent(3));
    const bool success = linearSolver_->solve(d_HscCSR_, d_bsc_.values(), d_xp_.values());
    if (!success)
    {
        return false;
    }

    // Solve linear equation about Δxl
    // Hll*Δxl = -bl - HplT*Δxp
    gpu::schurComplementPost(
        d_invHll_, d_bl_, d_Hpl_, d_xp_, d_xl_, cudaDevice_.getStreamAndEvent(3));
    
    profItems_[PROF_ITEM_SCHUR_COMPLEMENT] = cudaDevice_.stopTimingEvent();

    return true;
}

void BlockSolver::update(const VertexSetVec& vertexSets)
{
    cudaDevice_.startTimingEvent();

    for (BaseVertexSet* vertexSet : vertexSets)
    {
        if (!vertexSet->isMarginilised())
        {
            // TODO: Making the assumption here that this is a Se3D pose - it may not be.
            // This and the landmark estimate data need to be passed as void* (maybe?)
            PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSet);
            GpuVecSe3d& estimateData = poseVertexSet->getDeviceEstimates();
            gpu::updatePoses(d_xp_, estimateData, cudaDevice_.getStreamAndEvent(1));
        }
        else
        {
            LandmarkVertexSet* lmVertexSet = static_cast<LandmarkVertexSet*>(vertexSet);
            GpuVec3d& estimateData = lmVertexSet->getDeviceEstimates();
            gpu::updateLandmarks(d_xl_, estimateData, cudaDevice_.getStreamAndEvent(2));
        }
    }

    profItems_[PROF_ITEM_UPDATE] = cudaDevice_.stopTimingEvent();
}

double BlockSolver::computeScale(double lambda)
{
    gpu::computeScale(d_x_, d_b_, d_chi_, ScalarCast(lambda), cudaDevice_.getStreamAndEvent(1));
    Scalar scale = 0;
    d_chi_.download(&scale);
    return scale;
}

void BlockSolver::updateEdges(const EdgeSetVec& edgeSets)
{ 
    for (const auto& edgeSet : edgeSets)
    {
        edgeSet->updateEdges(cudaDevice_.getStreamAndEvent(1));
    }
}

void BlockSolver::push()
{
    d_solution_.copyToAsync(d_solutionBackup_, cudaDevice_.getStream(5));
}

void BlockSolver::pop() 
{ 
    d_solutionBackup_.copyToAsync(d_solution_, cudaDevice_.getStream(5)); 
}

void BlockSolver::finalize(const VertexSetVec& vertexSets)
{
    for (BaseVertexSet* vertexSet : vertexSets)
    {
        if (!vertexSet->isMarginilised())
        {
            PoseVertexSet* poseVertexSet = dynamic_cast<PoseVertexSet*>(vertexSet);
            assert(poseVertexSet != nullptr);
            poseVertexSet->finalise();
        }
        else
        {
            LandmarkVertexSet* lmVertexSet = dynamic_cast<LandmarkVertexSet*>(vertexSet);
            assert(lmVertexSet != nullptr);
            lmVertexSet->finalise();
        }
    }
}

void BlockSolver::clear() noexcept
{ 
    nedges_ = 0;
    numP_ = 0;
    numL_ = 0;
    verticesP_.clear();
    verticesL_.clear();
    HplblockPos_.clear();
}

void BlockSolver::getTimeProfile(TimeProfile& prof) const
{
    static const char* profileItemString[PROF_ITEM_COUNT] = {
        "0: Initialize Optimizer",
        "1: Build Structure",
        "2: Compute Error",
        "3: Build System",
        "4: Schur Complement",
        "5: Symbolic Decomposition",
        "6: Numerical Decomposition",
        "7: Update Solution",
        "8: Hpp linear solver"};

    prof.clear();
    for (int i = 0; i < PROF_ITEM_COUNT; i++)
    {
        prof[profileItemString[i]] = profItems_[i];
    }
}
} // namespace cugo