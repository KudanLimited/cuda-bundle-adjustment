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

template <class T>
using UniquePtr = std::unique_ptr<T>;

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

using BatchStatistics = std::vector<BatchInfo>;

/** @brief Time profile.
*/
using TimeProfile = std::map<std::string, double>;

////////////////////////////////////////////////////////////////////////////////////
// Cuda Bundle Adjustment
////////////////////////////////////////////////////////////////////////////////////
// forward declerations
struct PoseVertex;
struct LandmarkVertex;
struct CameraParams;
class BaseEdgeSet;
class CudaBlockSolver;
class CudaBundleAdjustmentImpl;

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

	/** @brief Clears the graph.
	*/
	virtual void clear() = 0;

	/** @brief Returns the batch statistics.
	*/
	virtual const BatchStatistics& batchStatistics() const = 0;

	/** @brief Returns the time profile.
	*/
	virtual const TimeProfile& timeProfile() = 0;

	/** @brief the destructor.
	*/
	virtual ~CudaBundleAdjustment();

	virtual void addPoseVertex(PoseVertex* v) = 0;
	virtual void addLandmarkVertex(LandmarkVertex* v) = 0;
	virtual PoseVertex* poseVertex(int id) const = 0;
	virtual LandmarkVertex* landmarkVertex(int id) const = 0;
	virtual bool removePoseVertex(BaseEdgeSet* edgeSet, PoseVertex* v) = 0;
	virtual bool removeLandmarkVertex(BaseEdgeSet* edgeSet, LandmarkVertex* v) = 0;
	virtual size_t nposes() const = 0;
	virtual size_t nlandmarks() const = 0;
	virtual std::array<BaseEdgeSet*, 6>& getEdgeSets() = 0;

	virtual void setCameraPrams(const CameraParams& camera) = 0;
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
		for(int i = 0; i < edgeSets.size(); ++i)
		{
			if (!edgeSets[i]) 
			{
				edgeSets[i] = edgeSet;
				return true;
			}
		}
		return false;
	}

	void addPoseVertex(PoseVertex* v) override;

	void addLandmarkVertex(LandmarkVertex* v) override;
	PoseVertex* poseVertex(int id) const override;

	LandmarkVertex* landmarkVertex(int id) const override;

	bool removePoseVertex(BaseEdgeSet* edgeSet, PoseVertex* v) override;

	bool removeLandmarkVertex(BaseEdgeSet* edgeSet, LandmarkVertex* v) override;

	size_t nposes() const override;

	size_t nlandmarks() const override;
	
	std::array<BaseEdgeSet*, 6>& getEdgeSets() override;

	void setCameraPrams(const CameraParams& camera) override;

	void initialize() override;

	void optimize(int niterations) override;

	void clear() override;

	const BatchStatistics& batchStatistics() const override;

	const TimeProfile& timeProfile() override;

	~CudaBundleAdjustmentImpl();

private:

	static inline double attenuation(double x) { return 1 - std::pow(2 * x - 1, 3); }
	static inline double clamp(double v, double lo, double hi) { return std::max(lo, std::min(v, hi)); }

	std::map<int, PoseVertex*> vertexMapP;  //!< connected pose vertices.
	std::map<int, LandmarkVertex*> vertexMapL; //!< connected landmark vertices.

	std::array<BaseEdgeSet*, 6> edgeSets = { nullptr };

	std::unique_ptr<CudaBlockSolver> solver_;
	std::unique_ptr<CameraParams> camera_;

	BatchStatistics stats_;
	TimeProfile timeProfile_;
};

} // namespace cuba

#endif // !__CUDA_BUNDLE_ADJUSTMENT_H__
