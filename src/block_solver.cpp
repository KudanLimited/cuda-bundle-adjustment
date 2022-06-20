#include "block_solver.h"

#include "cuda_linear_solver.h"
#include "optimisable_graph.h"
#include "profile.h"

namespace cugo
{

void BlockSolver::initialize(
    CameraParams* camera, const EdgeSetVec& edgeSets, const VertexSetVec& vertexSets)
{
    const auto t0 = get_time_point();

    for (BaseEdgeSet* edgeSet : edgeSets)
    {
        edgeSet->clear();
    }
    for (BaseVertexSet* vertexSet : vertexSets)
    {
        vertexSet->clear();

        if (vertexSet->isMarginilised())
        {
            // only perform schur if we have landmark vertices
            this->doSchur = true;
        }
    }

    // upload camera parameters to constant memory if defined
    if (camera)
    {
        std::vector<Scalar> cameraParams(5);
        cameraParams[0] = ScalarCast(camera->fx);
        cameraParams[1] = ScalarCast(camera->fy);
        cameraParams[2] = ScalarCast(camera->cx);
        cameraParams[3] = ScalarCast(camera->cy);
        cameraParams[4] = ScalarCast(camera->bf);
        gpu::setCameraParameters(cameraParams.data());
    }

    // create sparse linear solver
    if (!linearSolver_)
    {
        if (doSchur)
        {
            linearSolver_ = std::make_unique<HscSparseLinearSolver>();
        }
        else
        {
            linearSolver_ = std::make_unique<DenseLinearSolver>();
        }
    }

    profItems_.assign(PROF_ITEM_NUM, 0);

    const auto t1 = get_time_point();
    profItems_[PROF_ITEM_INITIALIZE] += get_duration(t0, t1);
}

void BlockSolver::buildStructure(const EdgeSetVec& edgeSets, const VertexSetVec& vertexSets)
{
    assert(edgeSets.size() > 0);
    assert(vertexSets.size() > 0);

    const auto t0 = get_time_point();

    size_t accumSizeL = 0;
    size_t accumSizeP = 0;
    std::for_each(
        vertexSets.begin(), vertexSets.end(), [&accumSizeL, &accumSizeP](BaseVertexSet* set) {
            if (set->isMarginilised())
            {
                accumSizeL += set->size();
            }
            else
            {
                accumSizeP += set->size();
            }
        });

    // calculate the solutions....
    // use one larget buffer to avoid having to upload
    // numerous vertex data buffers to the GPU
    d_solution_.resize(accumSizeL * 3 + accumSizeP * 7);
    d_solutionBackup_.resize(d_solution_.size());

    std::vector<BaseVertex*> verticesP;
    std::vector<BaseVertex*> verticesL;
    verticesP.reserve(accumSizeP);
    if (accumSizeL > 0)
    {
        verticesL.reserve(accumSizeL);
    }

    size_t numP = 0;
    size_t numL = 0;

    int offset = 0;
    for (BaseVertexSet* vertexSet : vertexSets)
    {
        if (!vertexSet->isMarginilised())
        {
            PoseVertexSet* poseVertexSet = dynamic_cast<PoseVertexSet*>(vertexSet);
            assert(poseVertexSet != nullptr);
            poseVertexSet->mapEstimateData(d_solution_.data() + offset);
            offset += poseVertexSet->getDeviceEstimateSize() * 7;

            auto& setData = poseVertexSet->get();
            verticesP.insert(verticesP.end(), setData.begin(), setData.end());
            numP += vertexSet->getActiveSize();
        }
        else
        {
            LandmarkVertexSet* lmVertexSet = dynamic_cast<LandmarkVertexSet*>(vertexSet);
            assert(lmVertexSet != nullptr);
            lmVertexSet->mapEstimateData(d_solution_.data() + offset);
            offset += lmVertexSet->getDeviceEstimateSize() * 3;

            auto& setData = lmVertexSet->get();
            verticesL.insert(verticesL.end(), setData.begin(), setData.end());
            numL += vertexSet->getActiveSize();
        }
    }

    // setup the edge set estimation data
    nedges_ = 0;
    size_t nVertexBlockPos = 0;
    int edgeId = 0;
    for (BaseEdgeSet* edgeSet : edgeSets)
    {
        edgeSet->init(edgeId, doSchur);
        nedges_ += edgeSet->nedges();
        if (doSchur)
        {
            nVertexBlockPos += edgeSet->getHessianBlockPosSize();
        }
    }

    if (doSchur)
    {
        // build Hpl block matrix structure
        std::vector<HplBlockPos> hplBlockPos;
        hplBlockPos.reserve(nVertexBlockPos);

        for (BaseEdgeSet* edgeSet : edgeSets)
        {
            auto& block = edgeSet->getHessianBlockPos();
            hplBlockPos.insert(hplBlockPos.end(), block.begin(), block.end());
        }

        d_HplBlockPos_.assign(nVertexBlockPos, hplBlockPos.data());
        d_Hpl_.resize(numP, numL);
        d_Hpl_.resizeNonZeros(d_HplBlockPos_.size());
        d_nnzPerCol_.resize(accumSizeL + 1);
        d_edge2Hpl_.resize(nedges_);

        gpu::buildHplStructure(d_HplBlockPos_, d_Hpl_, d_edge2Hpl_, d_nnzPerCol_);

        // build host Hschur sparse block matrix structure
        Hsc_.resize(numP, numP);
        Hsc_.constructFromVertices(verticesL);
        Hsc_.convertBSRToCSR();

        // initialise the device schur hessian matrix
        d_Hsc_.resize(numP, numP);
        d_Hsc_.resizeNonZeros(Hsc_.nblocks());
        d_Hsc_.upload(nullptr, Hsc_.outerIndices(), Hsc_.innerIndices());

        // initialise the device pose and landmark Hessian matrices
        // these are filled by the computation of the quadratic form
        d_Hll_.resize(numL);
        d_Hpp_.resize(numP);

        d_HscCSR_.resize(Hsc_.nnzSymm());
        d_BSR2CSR_.assign(Hsc_.nnzSymm(), (int*)Hsc_.BSR2CSR());

        d_HscMulBlockIds_.resize(Hsc_.nmulBlocks());
        gpu::findHschureMulBlockIndices(d_Hpl_, d_Hsc_, d_HscMulBlockIds_);

        d_bsc_.resize(numP);
        d_Hpl_invHll_.resize(nVertexBlockPos);
        d_HllBackup_.resize(numL);
        d_invHll_.resize(numL);
    }
    else
    {
        Hpp_.resize(numP, numP);
        Hpp_.constructFromVertices(verticesP);
        Hpp_.convertBSRToCSR();

        d_Hpp_.resize(numP);
    }

    // allocate device buffers
    d_x_.resize(numP * PDIM + accumSizeL * LDIM);
    d_b_.resize(numP * PDIM + accumSizeL * LDIM);

    d_xp_.map(numP, d_x_.data());
    d_bp_.map(numP, d_b_.data());

    if (doSchur)
    {
        d_xl_.map(numL, d_x_.data() + numP * PDIM);
        d_bl_.map(numL, d_b_.data() + numP * PDIM);
    }

    d_HppBackup_.resize(numP);

    // upload edge information to device memory
    int prevEdgeSize = 0;
    for (int i = 0; i < edgeSets.size(); ++i)
    {
        // upload the graph data to the device
        // Note: nullptr passed to map function if no landmark
        // data as the Hpl matrix then doesn't make sense.
        int* edge2HplPtr = nullptr;
        if (doSchur)
        {
            // data layout is pose data first followed by landmark
            edge2HplPtr = d_edge2Hpl_.data() + prevEdgeSize;
        }
        edgeSets[i]->mapDevice(edge2HplPtr);
        prevEdgeSize += edgeSets[i]->nedges();
    }

    d_chi_.resize(1);

    const auto t1 = get_time_point();

    if (doSchur)
    {
        HscSparseLinearSolver* sparseLinearSolver =
            static_cast<HscSparseLinearSolver*>(linearSolver_.get());

        // analyze pattern of Hschur matrix (symbolic decomposition)
        sparseLinearSolver->initialize(Hsc_);

        const auto t2 = get_time_point();
        profItems_[PROF_ITEM_DECOMP_SYMBOLIC] += get_duration(t1, t2);
    }
    else
    {
        DenseLinearSolver* sparseLinearSolver =
            static_cast<DenseLinearSolver*>(linearSolver_.get());
        sparseLinearSolver->initialize(Hpp_);

        const auto t2 = get_time_point();
        profItems_[PROF_ITEM_SOLVE_HPP] += get_duration(t1, t2);
    }

    profItems_[PROF_ITEM_BUILD_STRUCTURE] += get_duration(t0, t1);
}

double BlockSolver::computeErrors(const EdgeSetVec& edgeSets, const VertexSetVec& vertexSets)
{
    const auto t0 = get_time_point();

    maths::Vec3 rho;
    Scalar accumChi = 0;
    for (const EdgeSet* edgeSet : edgeSets)
    {
        const Scalar chi = edgeSet->computeError(vertexSets, d_chi_);
        if (edgeSet->kernel())
        {
            edgeSet->kernel()->robustify(chi, rho);
            accumChi += rho[0];
        }
        else
        {
            accumChi += chi;
        }
    }

    const auto t1 = get_time_point();
    profItems_[PROF_ITEM_COMPUTE_ERROR] += get_duration(t0, t1);

    return accumChi;
}

void BlockSolver::buildSystem(const EdgeSetVec& edgeSets, const VertexSetVec& vertexSets)
{
    const auto t0 = get_time_point();

    ////////////////////////////////////////////////////////////////////////////////////
    // Build linear system about solution increments Δx
    // H*Δx = -b
    //
    // coefficient matrix are divided into blocks, and each block is calculated
    // | Hpp  Hpl ||Δxp| = |-bp|
    // | HplT Hll ||Δxl|   |-bl|
    ////////////////////////////////////////////////////////////////////////////////////

    d_Hpp_.fillZero();
    d_bp_.fillZero();

    if (doSchur)
    {
        d_Hll_.fillZero();
        d_bl_.fillZero();
    }

    for (auto* edgeSet : edgeSets)
    {
        edgeSet->constructQuadraticForm(vertexSets, d_Hpp_, d_bp_, d_Hll_, d_bl_, d_Hpl_);
    }

    const auto t1 = get_time_point();
    profItems_[PROF_ITEM_BUILD_SYSTEM] += get_duration(t0, t1);
}

double BlockSolver::maxDiagonal()
{
    DeviceBuffer<Scalar> d_buffer(16);
    const Scalar maxP = gpu::maxDiagonal(d_Hpp_, d_buffer);
    if (doSchur)
    {
        const Scalar maxL = gpu::maxDiagonal(d_Hll_, d_buffer);
        return std::max(maxP, maxL);
    }
    return maxP;
}

void BlockSolver::setLambda(double lambda)
{
    gpu::addLambda(d_Hpp_, ScalarCast(lambda), d_HppBackup_);
    if (doSchur)
    {
        gpu::addLambda(d_Hll_, ScalarCast(lambda), d_HllBackup_);
    }
}

void BlockSolver::restoreDiagonal()
{
    gpu::restoreDiagonal(d_Hpp_, d_HppBackup_);
    if (doSchur)
    {
        gpu::restoreDiagonal(d_Hll_, d_HllBackup_);
    }
}

bool BlockSolver::solve()
{
    if (!doSchur)
    {
        const auto t1 = get_time_point();
        const bool success = linearSolver_->solve(d_Hpp_.values(), d_bp_.values(), d_xp_.values());
        const auto t2 = get_time_point();
        if (!success)
        {
            return false;
        }
        profItems_[PROF_ITEM_SOLVE_HPP] += get_duration(t1, t2);
        return true;
    }

    const auto t0 = get_time_point();

    ////////////////////////////////////////////////////////////////////////////////////
    // Schur complement
    // bSc = -bp + Hpl*Hll^-1*bl
    // HSc = Hpp - Hpl*Hll^-1*HplT
    ////////////////////////////////////////////////////////////////////////////////////
    gpu::computeBschure(d_bp_, d_Hpl_, d_Hll_, d_bl_, d_bsc_, d_invHll_, d_Hpl_invHll_);
    gpu::computeHschure(d_Hpp_, d_Hpl_invHll_, d_Hpl_, d_HscMulBlockIds_, d_Hsc_);

    const auto t1 = get_time_point();

    ////////////////////////////////////////////////////////////////////////////////////
    // Solve linear equation about Δxp
    // HSc*Δxp = bp
    ////////////////////////////////////////////////////////////////////////////////////
    gpu::convertHschureBSRToCSR(d_Hsc_, d_BSR2CSR_, d_HscCSR_);
    const bool success = linearSolver_->solve(d_HscCSR_, d_bsc_.values(), d_xp_.values());
    if (!success)
    {
        return false;
    }

    const auto t2 = get_time_point();

    ////////////////////////////////////////////////////////////////////////////////////
    // Solve linear equation about Δxl
    // Hll*Δxl = -bl - HplT*Δxp
    ////////////////////////////////////////////////////////////////////////////////////
    gpu::schurComplementPost(d_invHll_, d_bl_, d_Hpl_, d_xp_, d_xl_);

    const auto t3 = get_time_point();
    profItems_[PROF_ITEM_SCHUR_COMPLEMENT] += (get_duration(t0, t1) + get_duration(t2, t3));
    profItems_[PROF_ITEM_DECOMP_NUMERICAL] += get_duration(t1, t2);

    return true;
}

void BlockSolver::update(const VertexSetVec& vertexSets)
{
    const auto t0 = get_time_point();

    for (BaseVertexSet* vertexSet : vertexSets)
    {
        if (!vertexSet->isMarginilised())
        {
            // TODO: Making the assumption here that this is a Se3D pose - it may not be.
            // This and the landmark estimate data need to be passed as void* (maybe?)
            PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSet);
            GpuVecSe3d estimateData = poseVertexSet->getDeviceEstimates();
            gpu::updatePoses(d_xp_, estimateData);
        }
        else
        {
            LandmarkVertexSet* lmVertexSet = static_cast<LandmarkVertexSet*>(vertexSet);
            GpuVec3d estimateData = lmVertexSet->getDeviceEstimates();
            gpu::updateLandmarks(d_xl_, estimateData);
        }
    }

    const auto t1 = get_time_point();
    profItems_[PROF_ITEM_UPDATE] += get_duration(t0, t1);
}

double BlockSolver::computeScale(double lambda)
{
    gpu::computeScale(d_x_, d_b_, d_chi_, ScalarCast(lambda));
    Scalar scale = 0;
    d_chi_.download(&scale);
    return scale;
}

void BlockSolver::push() { d_solution_.copyTo(d_solutionBackup_); }

void BlockSolver::pop() { d_solutionBackup_.copyTo(d_solution_); }

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

void BlockSolver::getTimeProfile(TimeProfile& prof) const
{
    static const char* profileItemString[PROF_ITEM_NUM] = {
        "0: Initialize Optimizer",
        "1: Build Structure",
        "2: Compute Error",
        "3: Build System",
        "4: Schur Complement",
        "5: Symbolic Decomposition",
        "6: Numerical Decomposition",
        "7: Hpp linear solver (only for non-landmark optimistaion runs)",
        "8: Update Solution"};

    prof.clear();
    for (int i = 0; i < PROF_ITEM_NUM; i++)
    {
        prof[profileItemString[i]] = profItems_[i];
    }
}
} // namespace cugo