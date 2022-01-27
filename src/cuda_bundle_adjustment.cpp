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
#include "cuda_block_solver.h"
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

template <typename T>
static constexpr Scalar ScalarCast(T v) { return static_cast<Scalar>(v); }

void CudaBlockSolver::clear()
{
	verticesP_.clear();
	verticesL_.clear();
	baseEdges_.clear();
	HplBlockPos_.clear();
	qs_.clear();
	ts_.clear();
	Xws_.clear();
	omegas_.clear();
	edge2PL_.clear();
	edgeFlags_.clear();
}

void CudaBlockSolver::initialize(std::array<BaseEdgeSet*, 6>& edgeSets, std::map<int, PoseVertex*>& vertexMapP, 
	std::map<int, LandmarkVertex*>& vertexMapL, CameraParams* camera)
{
	assert(edgeSets.size() > 0);
	assert(camera != nullptr);

	const auto t0 = get_time_point();
	int totalEdgeSize = 0;

	for(const auto edgeSet : edgeSets)
	{
		if (!edgeSet)
		{
			break;
		}

		totalEdgeSize += edgeSet->nedges();
	}

	nedges_ = totalEdgeSize;
	clear();

	verticesP_.reserve(vertexMapP.size());
	verticesL_.reserve(vertexMapL.size());
	baseEdges_.reserve(totalEdgeSize);
	HplBlockPos_.reserve(totalEdgeSize);
	qs_.reserve(vertexMapP.size());
	ts_.reserve(vertexMapP.size());
	Xws_.reserve(vertexMapL.size());
	omegas_.reserve(totalEdgeSize);
	edge2PL_.reserve(totalEdgeSize);
	edgeFlags_.reserve(totalEdgeSize);

	std::vector<PoseVertex*> fixedVerticesP_;
	std::vector<LandmarkVertex*> fixedVerticesL_;
	int numP = 0;
	int numL = 0;

	// assign pose vertex id
	// gather rotations and translations into each vector
	for (const auto& [id, poseVertex] : vertexMapP)
	{
		if (!poseVertex->fixed)
		{
			poseVertex->iP = numP++;
			verticesP_.push_back(poseVertex);
			qs_.emplace_back(poseVertex->q.coeffs().data());
			ts_.emplace_back(poseVertex->t.data());
		}
		else
		{
			fixedVerticesP_.push_back(poseVertex);
		}
	}

	// assign landmark vertex id
	// gather 3D positions into vector
	for (const auto& [id, landmarkVertex] : vertexMapL)
	{
		if (!landmarkVertex->fixed)
		{
			landmarkVertex->iL = numL++;
			verticesL_.push_back(landmarkVertex);
			Xws_.emplace_back(landmarkVertex->Xw.data());
		}
		else
		{
			fixedVerticesL_.push_back(landmarkVertex);
		}
	}

	numP_ = numP;
	numL_ = numL;

	// inactive(fixed) vertices are added after active vertices
	for (auto poseVertex : fixedVerticesP_)
	{
		poseVertex->iP = numP++;
		verticesP_.push_back(poseVertex);
		qs_.emplace_back(poseVertex->q.coeffs().data());
		ts_.emplace_back(poseVertex->t.data());
	}

	for (auto landmarkVertex : fixedVerticesL_)
	{
		landmarkVertex->iL = numL++;
		verticesL_.push_back(landmarkVertex);
		Xws_.emplace_back(landmarkVertex->Xw.data());
	}

	// gather each edge members into each vector
	int edgeId = 0;
	for (const auto* edgeSet : edgeSets)
	{
		if (!edgeSet)
		{
			break;
		}

		for (auto* edge : edgeSet->get())
		{
			const auto poseVertex = edge->getPoseVertex();
			const auto landmarkVertex = edge->getLandmarkVertex();
			baseEdges_.push_back(edge);

			if (!poseVertex->fixed && !landmarkVertex->fixed)
			{
				HplBlockPos_.push_back({ poseVertex->iP, landmarkVertex->iL, edgeId });
			}

			omegas_.push_back(ScalarCast(edge->information()));
			edge2PL_.push_back({ poseVertex->iP, landmarkVertex->iL });
			edgeFlags_.push_back(makeEdgeFlag(poseVertex->fixed, landmarkVertex->fixed));

			edgeId++;
		}
	}

	nHplBlocks_ = static_cast<int>(HplBlockPos_.size());

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

void CudaBlockSolver::buildStructure(const std::array<BaseEdgeSet*, 6>& edgeSets)
{
	const auto t0 = get_time_point();

	// build Hpl block matrix structure
	d_Hpl_.resize(numP_, numL_);
	d_Hpl_.resizeNonZeros(nHplBlocks_);

	d_HplBlockPos_.assign(nHplBlocks_, HplBlockPos_.data());
	d_nnzPerCol_.resize(numL_ + 1);
	d_edge2Hpl_.resize(baseEdges_.size());

	gpu::buildHplStructure(d_HplBlockPos_, d_Hpl_, d_edge2Hpl_, d_nnzPerCol_);

	// build Hschur block matrix structure
	Hsc_.resize(numP_, numP_);
	Hsc_.constructFromVertices(verticesL_);
	Hsc_.convertBSRToCSR();

	d_Hsc_.resize(numP_, numP_);
	d_Hsc_.resizeNonZeros(Hsc_.nblocks());
	d_Hsc_.upload(nullptr, Hsc_.outerIndices(), Hsc_.innerIndices());

	d_HscCSR_.resize(Hsc_.nnzSymm());
	d_BSR2CSR_.assign(Hsc_.nnzSymm(), (int*)Hsc_.BSR2CSR());

	d_HscMulBlockIds_.resize(Hsc_.nmulBlocks());
	gpu::findHschureMulBlockIndices(d_Hpl_, d_Hsc_, d_HscMulBlockIds_);

	// allocate device buffers
	d_x_.resize(numP_ * PDIM + numL_ * LDIM);
	d_b_.resize(numP_ * PDIM + numL_ * LDIM);

	d_xp_.map(numP_, d_x_.data());
	d_bp_.map(numP_, d_b_.data());
	d_xl_.map(numL_, d_x_.data() + numP_ * PDIM);
	d_bl_.map(numL_, d_b_.data() + numP_ * PDIM);

	d_Hpp_.resize(numP_);
	d_Hll_.resize(numL_);

	d_HppBackup_.resize(numP_);
	d_HllBackup_.resize(numL_);

	d_bsc_.resize(numP_);
	d_invHll_.resize(numL_);
	d_Hpl_invHll_.resize(nHplBlocks_);

	// upload solutions to device memory
	d_solution_.resize(verticesP_.size() * 7 + verticesL_.size() * 3);
	d_solutionBackup_.resize(d_solution_.size());

	d_qs_.map(qs_.size(), d_solution_.data());
	d_ts_.map(ts_.size(), d_qs_.data() + d_qs_.size());
	d_Xws_.map(Xws_.size(), d_ts_.data() + d_ts_.size());
	
	d_qs_.upload(qs_.data());
	d_ts_.upload(ts_.data());
	d_Xws_.upload(Xws_.data());

	// upload edge information to device memory
	size_t index = 0;
	for (int i = 0; i < edgeSets.size(); ++i)
	{
		if (!edgeSets[i])
		{
			break;
		}

		BaseEdgeSet* edgeSet = edgeSets[i];
		size_t nedges = edgeSet->nedges();

		if (i > 0)
		{
			index += edgeSets[i - 1]->nedges();
		}

		// add the graph data to the device
		edgeSets[i]->initDevice(nedges, edgeSet->getMeasurementData(), omegas_.data() + index, 
			edge2PL_.data() + index, edgeFlags_.data() + index, d_edge2Hpl_.data() + index);
	}

	d_chi_.resize(1);

	const auto t1 = get_time_point();

	// analyze pattern of Hschur matrix (symbolic decomposition)
	linearSolver_->initialize(Hsc_);

	const auto t2 = get_time_point();

	profItems_[PROF_ITEM_BUILD_STRUCTURE] += get_duration(t0, t1);
	profItems_[PROF_ITEM_DECOMP_SYMBOLIC] += get_duration(t1, t2);
}

double CudaBlockSolver::computeErrors(const std::array<BaseEdgeSet*, 6>& edgeSets)
{
	const auto t0 = get_time_point();

	Scalar accumChi = 0;
	for (int i = 0; i < edgeSets.size(); ++i)
	{
		if (!edgeSets[i])
		{
			break;
		}

		const Scalar chi = edgeSets[i]->computeError(d_qs_, d_ts_, d_Xws_, d_chi_);
		accumChi += chi;
	}

	const auto t1 = get_time_point();
	profItems_[PROF_ITEM_COMPUTE_ERROR] += get_duration(t0, t1);

	return accumChi;
}

void CudaBlockSolver::buildSystem(const std::array<BaseEdgeSet*, 6>& edgeSets)
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
		if (!edgeSet)
		{
			break;
		}

		edgeSet->constructQuadraticForm(d_qs_, d_Hpp_, d_bp_, d_Hll_, d_bl_, d_Hpl_);
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
		return false;

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

void CudaBlockSolver::update()
{
	const auto t0 = get_time_point();

	gpu::updatePoses(d_xp_, d_qs_, d_ts_);
	gpu::updateLandmarks(d_xl_, d_Xws_);

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

void CudaBlockSolver::finalize()
{
	d_qs_.download(qs_.data());
	d_ts_.download(ts_.data());
	d_Xws_.download(Xws_.data());

	for (size_t i = 0; i < verticesP_.size(); i++)
	{
		qs_[i].copyTo(verticesP_[i]->q.coeffs().data());
		ts_[i].copyTo(verticesP_[i]->t.data());
	}

	for (size_t i = 0; i < verticesL_.size(); i++)
		Xws_[i].copyTo(verticesL_[i]->Xw.data());
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

inline uint8_t CudaBlockSolver::makeEdgeFlag(bool fixedP, bool fixedL)
{
	uint8_t flag = 0;
	if (fixedP) flag |= EDGE_FLAG_FIXED_P;
	if (fixedL) flag |= EDGE_FLAG_FIXED_L;
	return flag;
}

// CudaBundleAdjustmentImpl functions

void CudaBundleAdjustmentImpl::addPoseVertex(PoseVertex* v) 
{
	vertexMapP.insert({ v->id, v });
}

void CudaBundleAdjustmentImpl::addLandmarkVertex(LandmarkVertex* v) 
{
	vertexMapL.insert({ v->id, v });
}

PoseVertex* CudaBundleAdjustmentImpl::poseVertex(int id) const 
{
	auto it = vertexMapP.find(id);
	if (it == std::end(vertexMapP)) 
	{
		printf("Warning: id: %d not found in vertex map.\n");
		return nullptr;
	}
	return vertexMapP.at(id);
}

LandmarkVertex* CudaBundleAdjustmentImpl::landmarkVertex(int id) const 
{
	auto it = vertexMapL.find(id);
	if (it == std::end(vertexMapL)) 
	{
		printf("Warning: id: %d not found in Landmark map.\n");
		return nullptr;
	}
	return vertexMapL.at(id);
}

bool CudaBundleAdjustmentImpl::removePoseVertex(BaseEdgeSet* edgeSet, PoseVertex* v) 
{
	auto it = vertexMapP.find(v->id);
	if (it == std::end(vertexMapP))
	{
		return false;
	}

	for (auto e : it->second->edges)
	{
		edgeSet->removeEdge(e);
	}

	vertexMapP.erase(it);
	return true;
}

bool CudaBundleAdjustmentImpl::removeLandmarkVertex(BaseEdgeSet* edgeSet, LandmarkVertex* v) 
{
	auto it = vertexMapL.find(v->id);
	if (it == std::end(vertexMapL))
	{
		return false;
	}

	for (auto e : it->second->edges)
	{
		edgeSet->removeEdge(e);
	}

	vertexMapL.erase(it);
	return true;
}

size_t CudaBundleAdjustmentImpl::nposes() const 
{
	return vertexMapP.size();
}

size_t CudaBundleAdjustmentImpl::nlandmarks() const 
{
	return vertexMapL.size();
}

std::array<BaseEdgeSet*, 6>& CudaBundleAdjustmentImpl::getEdgeSets() 
{
	return edgeSets;
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
	solver_->initialize(edgeSets, vertexMapP, vertexMapL, camera_.get());
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
			solver_->buildStructure(edgeSets);
		}

		const double iniF = solver_->computeErrors(edgeSets);
		F = iniF;

		solver_->buildSystem(edgeSets);
		
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

			solver_->update();

			const double Fhat = solver_->computeErrors(edgeSets);
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

	solver_->finalize();

	solver_->getTimeProfile(timeProfile_);
}

void CudaBundleAdjustmentImpl::clear() 
{
	vertexMapL.clear();
	vertexMapP.clear();
	
	for (int i = 0; i < edgeSets.size(); ++i)
	{
		edgeSets[i] = nullptr;
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
