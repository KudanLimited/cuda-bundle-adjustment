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

#ifndef __CUDA_BUNDLE_ADJUSTMENT_H__
#define __CUDA_BUNDLE_ADJUSTMENT_H__

#include <memory>
#include <vector>
#include <map>
#include <array>
#include <string>
#include <cmath>

#include "device_matrix.h"
#include "device_buffer.h"


namespace cuba
{

// forward declerations
struct CameraParams;
class BaseEdgeSet;
class CudaBlockSolver;
class CudaBundleAdjustmentImpl;
class BaseVertexSet;
class BaseVertex;


template <class T>
using UniquePtr = std::unique_ptr<T>;

using EdgeSetVec = std::vector<BaseEdgeSet*>;
using VertexSetVec = std::vector<BaseVertexSet*>;

////////////////////////////////////////////////////////////////////////////////////
// Statistics
////////////////////////////////////////////////////////////////////////////////////

/** @brief information about optimization.
*/
struct BatchInfo
{
	int iteration;           //!< iteration number
	double chi2;             //!< total chi2 (objective function value)
};

class BatchStatistics
{
public:

	BatchInfo& getStartStats() { assert(!stats.empty()); return stats[0]; }
	BatchInfo& getLastStats() { assert(!stats.empty()); return stats.back(); }
	BatchInfo& getStatEntry(const int idx) { assert(stats.size() < idx); return stats[idx]; }
	void addStat(const BatchInfo& batchInfo) { stats.emplace_back(batchInfo); }
	const std::vector<BatchInfo>& get() { return stats; }
	void clear() { stats.clear(); }

private:

	std::vector<BatchInfo> stats;
};

/** @brief Time profile.
*/
using TimeProfile = std::map<std::string, double>;

template <typename T>
static constexpr Scalar ScalarCast(T v) { return static_cast<Scalar>(v); }

////////////////////////////////////////////////////////////////////////////////////
// Cuda Bundle Adjustment
////////////////////////////////////////////////////////////////////////////////////

/** @brief CUDA implementation of Bundle Adjustment.

The class implements a Bundle Adjustment algorithm with CUDA.
It optimizes camera poses and landmarks (3D points) represented by a graph.

@attention This class doesn't take responsibility for deleting pointers to vertices and edges
added in the graph.

*/
class CudaBundleAdjustment
{
public:

	using Ptr = UniquePtr<CudaBundleAdjustmentImpl>;

	/** @brief Creates an instance of CudaBundleAdjustment.
	*/
	static Ptr create();

	/** @brief Initializes the graph.
	*/
	virtual void initialize() = 0;

	/** @brief Optimizes the graph.
	@param niterations number of iterations for Levenberg-Marquardt algorithm.
	*/
	virtual void optimize(int niterations) = 0;

	/** @brief Returns the batch statistics.
	*/
	virtual BatchStatistics& batchStatistics() = 0;

	/** @brief Returns the time profile.
	*/
	virtual const TimeProfile& timeProfile() = 0;

	/** @brief the destructor.
	*/
	virtual ~CudaBundleAdjustment();

	virtual EdgeSetVec& getEdgeSets() = 0;

	virtual void setCameraPrams(const CameraParams& camera) = 0;

	virtual size_t nVertices(const int id) = 0;

	virtual void clearEdgeSets() = 0;
	virtual void clearVertexSets() = 0;
};

/** @brief Implementation of CudaBundleAdjustment.
*/
class CudaBundleAdjustmentImpl : public CudaBundleAdjustment
{
public:

	/**
	 * @brief constructor
	 */
	CudaBundleAdjustmentImpl();

	/** @brief adds a new graph to the optimiser with a custom edge 
	*/
	template <typename T>
	bool addEdgeSet(T* edgeSet)
	{
		assert(edgeSet != nullptr);
		edgeSets.push_back(edgeSet);
		return true;
	}

	template <typename T>
	bool addVertexSet(T* vertexSet)
	{
		assert(vertexSet != nullptr);
		vertexSets.push_back(vertexSet);
		return true;
	}
	
	EdgeSetVec& getEdgeSets() override;

	void setCameraPrams(const CameraParams& camera) override;

	void initialize() override;

	void optimize(int niterations) override;

	BatchStatistics& batchStatistics() override;

	const TimeProfile& timeProfile() override;

	size_t nVertices(const int id) override;

	void clearEdgeSets() override;
	void clearVertexSets() override;

	~CudaBundleAdjustmentImpl();

private:

	static inline double attenuation(double x) { return 1 - std::pow(2 * x - 1, 3); }
	static inline double clamp(double v, double lo, double hi) { return std::max(lo, std::min(v, hi)); }

	VertexSetVec vertexSets;
	EdgeSetVec edgeSets;

	std::unique_ptr<CudaBlockSolver> solver_;
	std::unique_ptr<CameraParams> camera_;

	BatchStatistics stats_;
	TimeProfile timeProfile_;
};

} // namespace cuba

#endif // !__CUDA_BUNDLE_ADJUSTMENT_H__
