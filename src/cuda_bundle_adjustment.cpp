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

#include "cuda_bundle_adjustment.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

#include "constants.h"
#include "sparse_block_matrix.h"
#include "device_buffer.h"
#include "device_matrix.h"
#include "cuda/cuda_block_solver.h"
#include "cuda_linear_solver.h"
#include "optimisable_graph.h"
#include "cuda_block_solver_impl.h"

namespace cuba
{

using VertexMapP = std::map<int, PoseVertex*>;
using VertexMapL = std::map<int, LandmarkVertex*>;
using EdgeSet2D = std::unordered_set<Edge2D*>;
using EdgeSet3D = std::unordered_set<Edge3D*>;
using time_point = decltype(std::chrono::steady_clock::now());

static inline time_point get_time_point()
{
	gpu::waitForKernelCompletion();
	return std::chrono::steady_clock::now();
}

static inline double get_duration(const time_point& from, const time_point& to)
{
	return std::chrono::duration_cast<std::chrono::duration<double>>(to - from).count();
}

void CudaBlockSolver::initialize(CameraParams* camera, const EdgeSetVec& edgeSets, const VertexSetVec& vertexSets)
{
	assert(camera != nullptr);

	const auto t0 = get_time_point();

	for (auto* edgeSet : edgeSets) 
	{
		edgeSet->clear();
	}
	for (auto* vertexSet : vertexSets) 
	{
		vertexSet->clear();
	}

	// upload camera parameters to constant memory
	std::vector<Scalar> cameraParams(5);
	cameraParams[0] = ScalarCast(camera->fx);
	cameraParams[1] = ScalarCast(camera->fy);
	cameraParams[2] = ScalarCast(camera->cx);
	cameraParams[3] = ScalarCast(camera->cy);
	cameraParams[4] = ScalarCast(camera->bf);
	gpu::setCameraParameters(cameraParams.data());

	// create sparse linear solver
	if (!linearSolver_)
	{
		linearSolver_ = SparseLinearSolver::create();
	}

	profItems_.assign(PROF_ITEM_NUM, 0);

	const auto t1 = get_time_point();
	profItems_[PROF_ITEM_INITIALIZE] += get_duration(t0, t1);
}

void CudaBlockSolver::buildStructure(const EdgeSetVec& edgeSets, const VertexSetVec& vertexSets)
{
	assert(edgeSets.size() > 0);
	assert(vertexSets.size() > 0);

	const auto t0 = get_time_point();

	size_t accumSizeL = 0;
	size_t accumSizeP = 0;
	std::for_each(vertexSets.begin(), vertexSets.end(), [&accumSizeL, &accumSizeP](BaseVertexSet* set) { 
		if(set->isMarginilised()) {
		accumSizeL += set->size(); 
	} else {
		accumSizeP += set->size(); 
	}});
	
	// calculate the solutions....
	// use one larget buffer to avoid having to upload
	// numerous vertex data buffers to the GPU
	d_solution_.resize(accumSizeL * 3 + accumSizeP * 7);
	d_solutionBackup_.resize(d_solution_.size());

	std::vector<BaseVertex*> verticesP;
	std::vector<BaseVertex*> verticesL;
	verticesL.reserve(accumSizeL);
	verticesP.reserve(accumSizeP);

	size_t numP = 0;
	size_t numL = 0;

	int offset = 0;
	for (auto* vertexSet : vertexSets)
	{
		assert(vertexSet != nullptr);
		if (!vertexSet->isMarginilised())
		{
			PoseVertexSet* poseVertexSet = static_cast<PoseVertexSet*>(vertexSet);
			poseVertexSet->mapEstimateData(d_solution_.data() + offset);
			offset += poseVertexSet->getDeviceEstimateSize() * 7;
			auto& setData = poseVertexSet->get();			
			verticesP.insert(verticesP.end(), setData.begin(), setData.end());
			numP += vertexSet->getActiveSize();
		}
		else
		{
			LandmarkVertexSet* lmVertexSet = static_cast<LandmarkVertexSet*>(vertexSet);
			lmVertexSet->mapEstimateData(d_solution_.data() + offset);
			offset += lmVertexSet->getDeviceEstimateSize() * 3;

			auto& setData = lmVertexSet->get();	
			verticesL.insert(verticesL.end(), setData.begin(), setData.end());
			numL += vertexSet->getActiveSize();
		}
	}

	// setup the edge set estimation data
	int nedges = 0;
	size_t nVertexBlockPos = 0;
	int edgeId = 0;
	for (auto* edgeSet : edgeSets)
	{
		edgeSet->init(edgeId); 
		nVertexBlockPos += edgeSet->getHessianBlockPosSize(); 
		nedges += edgeSet->nedges(); 
	}

	// build Hpl block matrix structure	
	std::vector<HplBlockPos> hplBlockPos;
	hplBlockPos.reserve(nVertexBlockPos);

	for (auto* edgeSet : edgeSets)
	{
		auto& block = edgeSet->getHessianBlockPos();
		hplBlockPos.insert(hplBlockPos.end(), block.begin(), block.end());
	}
	
	d_HplBlockPos_.assign(nVertexBlockPos, hplBlockPos.data());
	d_Hpl_.resize(numP, numL);
	d_Hpl_.resizeNonZeros(d_HplBlockPos_.size());
	d_nnzPerCol_.resize(numL + 1);
	d_edge2Hpl_.resize(nedges);

	gpu::buildHplStructure(d_HplBlockPos_, d_Hpl_, d_edge2Hpl_, d_nnzPerCol_);

	// build Hschur block matrix structure
	Hsc_.resize(numP, numP);
	Hsc_.constructFromVertices(verticesL);
	Hsc_.convertBSRToCSR();

	d_Hsc_.resize(numP, numP);
	d_Hsc_.resizeNonZeros(Hsc_.nblocks());
	d_Hsc_.upload(nullptr, Hsc_.outerIndices(), Hsc_.innerIndices());

	d_HscCSR_.resize(Hsc_.nnzSymm());
	d_BSR2CSR_.assign(Hsc_.nnzSymm(), (int*)Hsc_.BSR2CSR());

	d_HscMulBlockIds_.resize(Hsc_.nmulBlocks());
	gpu::findHschureMulBlockIndices(d_Hpl_, d_Hsc_, d_HscMulBlockIds_);

	// allocate device buffers
	d_x_.resize(numP * PDIM + numL * LDIM);
	d_b_.resize(numP * PDIM + numL * LDIM);

	d_xp_.map(numP, d_x_.data());
	d_bp_.map(numP, d_b_.data());
	d_xl_.map(numL, d_x_.data() + numP * PDIM);
	d_bl_.map(numL, d_b_.data() + numP * PDIM);

	d_Hpp_.resize(numP);
	d_Hll_.resize(numL);

	d_HppBackup_.resize(numP);
	d_HllBackup_.resize(numL);

	d_bsc_.resize(numP);
	d_invHll_.resize(numL);
	d_Hpl_invHll_.resize(nVertexBlockPos);

	// upload edge information to device memory
	int prevEdgeSize = 0;
	for (int i = 0; i < edgeSets.size(); ++i)
	{
		// upload the graph data to the device
		edgeSets[i]->mapDevice(d_edge2Hpl_.data() + prevEdgeSize);
		prevEdgeSize += edgeSets[i]->nedges();
	}

	d_chi_.resize(1);

	const auto t1 = get_time_point();

	// analyze pattern of Hschur matrix (symbolic decomposition)
	linearSolver_->initialize(Hsc_);

	const auto t2 = get_time_point();

	profItems_[PROF_ITEM_BUILD_STRUCTURE] += get_duration(t0, t1);
	profItems_[PROF_ITEM_DECOMP_SYMBOLIC] += get_duration(t1, t2);
}

double CudaBlockSolver::computeErrors(const EdgeSetVec& edgeSets, const VertexSetVec& vertexSets)
{
	const auto t0 = get_time_point();

	Scalar accumChi = 0;
	for (int i = 0; i < edgeSets.size(); ++i)
	{
		const Scalar chi = edgeSets[i]->computeError(vertexSets, d_chi_);
		accumChi += chi;
	}

	const auto t1 = get_time_point();
	profItems_[PROF_ITEM_COMPUTE_ERROR] += get_duration(t0, t1);

	return accumChi;
}

void CudaBlockSolver::buildSystem(const EdgeSetVec& edgeSets, const VertexSetVec& vertexSets)
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
	d_Hll_.fillZero();
	d_bp_.fillZero();
	d_bl_.fillZero();

	for (auto* edgeSet : edgeSets)
	{
		edgeSet->constructQuadraticForm(vertexSets, d_Hpp_, d_bp_, d_Hll_, d_bl_, d_Hpl_);
	}

	const auto t1 = get_time_point();
	profItems_[PROF_ITEM_BUILD_SYSTEM] += get_duration(t0, t1);
}

double CudaBlockSolver::maxDiagonal()
{
	DeviceBuffer<Scalar> d_buffer(16);
	const Scalar maxP = gpu::maxDiagonal(d_Hpp_, d_buffer);
	const Scalar maxL = gpu::maxDiagonal(d_Hll_, d_buffer);
	return std::max(maxP, maxL);
}

void CudaBlockSolver::setLambda(double lambda)
{
	gpu::addLambda(d_Hpp_, ScalarCast(lambda), d_HppBackup_);
	gpu::addLambda(d_Hll_, ScalarCast(lambda), d_HllBackup_);
}

void CudaBlockSolver::restoreDiagonal()
{
	gpu::restoreDiagonal(d_Hpp_, d_HppBackup_);
	gpu::restoreDiagonal(d_Hll_, d_HllBackup_);
}

bool CudaBlockSolver::solve()
{
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

void CudaBlockSolver::update(const VertexSetVec& vertexSets)
{
	const auto t0 = get_time_point();

	for (auto* vertexSet : vertexSets)
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

double CudaBlockSolver::computeScale(double lambda)
{
	gpu::computeScale(d_x_, d_b_, d_chi_, ScalarCast(lambda));
	Scalar scale = 0;
	d_chi_.download(&scale);
	return scale;
}

void CudaBlockSolver::push()
{
	d_solution_.copyTo(d_solutionBackup_);
}

void CudaBlockSolver::pop()
{
	d_solutionBackup_.copyTo(d_solution_);
}

void CudaBlockSolver::finalize(const VertexSetVec& vertexSets)
{
	for (auto* vertexSet : vertexSets)
	{
		vertexSet->finalise();
	}
}

void CudaBlockSolver::getTimeProfile(TimeProfile& prof) const
{
	static const char* profileItemString[PROF_ITEM_NUM] =
	{
		"0: Initialize Optimizer",
		"1: Build Structure",
		"2: Compute Error",
		"3: Build System",
		"4: Schur Complement",
		"5: Symbolic Decomposition",
		"6: Numerical Decomposition",
		"7: Update Solution"
	};

	prof.clear();
	for (int i = 0; i < PROF_ITEM_NUM; i++)
		prof[profileItemString[i]] = profItems_[i];
}

// CudaBundleAdjustmentImpl functions

EdgeSetVec& CudaBundleAdjustmentImpl::getEdgeSets() 
{
	return edgeSets;
}

size_t CudaBundleAdjustmentImpl::nVertices(const int id)
{
	assert(id < vertexSets.size());
	return vertexSets[id]->size();
}

void CudaBundleAdjustmentImpl::setCameraPrams(const CameraParams& camera)
{
	if (camera_)
	{
		camera_ = nullptr;
	}
	camera_ = std::make_unique<CameraParams>(camera);
}

void CudaBundleAdjustmentImpl::initialize() 
{
	solver_->initialize(camera_.get(), edgeSets, vertexSets);
	stats_.clear();
}

void CudaBundleAdjustmentImpl::optimize(int niterations) 
{
	const int maxq = 10;
	const double tau = 1e-5;

	double nu = 2;
	double lambda = 0;
	double F = 0;

	// Levenberg-Marquardt iteration
	for (int iteration = 0; iteration < niterations; iteration++)
	{
		if (iteration == 0)
		{
			solver_->buildStructure(edgeSets, vertexSets);
		}

		const double iniF = solver_->computeErrors(edgeSets, vertexSets);
		F = iniF;

		solver_->buildSystem(edgeSets, vertexSets);
		
		if (iteration == 0)
		{
			lambda = tau * solver_->maxDiagonal();
		}

		int q = 0;
		double rho = -1;
		for (; q < maxq && rho < 0; q++)
		{
			solver_->push();

			solver_->setLambda(lambda);

			const bool success = solver_->solve();

			solver_->update(vertexSets);

			const double Fhat = solver_->computeErrors(edgeSets, vertexSets);
			const double scale = solver_->computeScale(lambda) + 1e-3;
			rho = success ? (F - Fhat) / scale : -1;

			if (rho > 0)
			{
				lambda *= clamp(attenuation(rho), 1./3, 2./3);
				nu = 2;
				F = Fhat;
				break;
			}
			else
			{
				lambda *= nu;
				nu *= 2;
				solver_->restoreDiagonal();
				solver_->pop();
			}
		}

		stats_.push_back({ iteration, F });

		if (q == maxq || rho <= 0 || !std::isfinite(lambda))
		{
			break;
		}
	}

	solver_->finalize(vertexSets);

	solver_->getTimeProfile(timeProfile_);
}

void CudaBundleAdjustmentImpl::clear() 
{	
	for (auto* edgeSet : edgeSets)
	{
		for (auto* edge : edgeSet->get())
		{
			delete edge;
			edge = nullptr;
		}
		delete edgeSet;
		edgeSet = nullptr;
	}
	for (auto* vertexSet : vertexSets)
	{
		for (auto* vertex : vertexSet->get())
		{
			delete vertex;
			vertex = nullptr;
		}
		delete vertexSet;
		vertexSet = nullptr;
	}

}

const BatchStatistics& CudaBundleAdjustmentImpl::batchStatistics() const 
{
	return stats_;
}

const TimeProfile& CudaBundleAdjustmentImpl::timeProfile()
{
	return timeProfile_;
}

CudaBundleAdjustmentImpl::CudaBundleAdjustmentImpl() :
	solver_(std::make_unique<CudaBlockSolver>()), camera_(std::make_unique<CameraParams>())
{}

CudaBundleAdjustmentImpl::~CudaBundleAdjustmentImpl()
{
	clear();
}

CudaBundleAdjustment::Ptr CudaBundleAdjustment::create()
{
	return std::make_unique<CudaBundleAdjustmentImpl>();
}

CudaBundleAdjustment::~CudaBundleAdjustment()
{
}

} // namespace cuba
