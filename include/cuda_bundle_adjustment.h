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

#include "cuda_bundle_adjustment_types.h"

namespace cuba
{

/** @brief CUDA implementation of Bundle Adjustment.

The class implements a Bundle Adjustment algorithm with CUDA.
It optimizes camera poses and landmarks (3D points) represented by a graph.

@attention This class doesn't take responsibility for deleting pointers to vertices and edges
added in the graph.

*/
class CudaBundleAdjustment
{
public:

	using Ptr = UniquePtr<CudaBundleAdjustment>;

	/** @brief Creates an instance of CudaBundleAdjustment.
	*/
	static Ptr create();

	/** @brief adds a new graph to the optimiser with a custom edge 
	*/
	template <typename T>
	bool addEdgeSet(std::unique_ptr<T>& edgeSet)
	{
		for(int i = 0; i < edgeSets.size(); ++i)
		{
			if (!edgeSets[i]) 
			{
				edgeSets[i] = std::move(edgeSet);
				return true;
			}
		}
		return false;
	}

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

	void addPoseVertex(VertexP* v) override;
	void addLandmarkVertex(VertexP* v) override;
	VertexP* poseVertex(int id) const override;
	VertexL* landmarkVertex(int id) const override;
	void removePoseVertex(PoseVertex* v) override;
	void removeLandmarkVertex(PoseVertex* v) override;
	size_t nposes() const override;
	size_t nlandmarks() const override;
	void clear() override;

};

} // namespace cuba

#endif // !__CUDA_BUNDLE_ADJUSTMENT_H__
