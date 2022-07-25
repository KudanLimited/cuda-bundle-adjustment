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

#ifndef __CUDA_GRAPH_OPTIMISATION_H__
#define __CUDA_GRAPH_OPTIMISATION_H__

#include "device_buffer.h"
#include "device_matrix.h"
#include "macro.h"

#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>


namespace cugo
{
// forward declerations
struct CameraParams;
class BaseEdgeSet;
class BlockSolver;
class CudaGraphOptimisationImpl;
class BaseVertexSet;
class BaseVertex;


template <class T>
using UniquePtr = std::unique_ptr<T>;

using EdgeSetVec = std::vector<BaseEdgeSet*>;
using VertexSetVec = std::vector<BaseVertexSet*>;

////////////////////////////////////////////////////////////////////////////////////
// Camera parameters
////////////////////////////////////////////////////////////////////////////////////

/** @brief Camera parameters struct.
 */
struct CUGO_API CameraParams
{
    double fx; //!< focal length x (pixel)
    double fy; //!< focal length y (pixel)
    double cx; //!< principal point x (pixel)
    double cy; //!< principal point y (pixel)
    double bf; //!< stereo baseline times fx

    /** @brief The default constructor.
     */
    CameraParams() : fx(0.0), fy(0.0), cx(0.0), cy(0.0), bf(0.0) {}

    CameraParams(double fx, double fy, double cx, double cy, double bf)
        : fx(fx), fy(fy), cx(cx), cy(cy), bf(bf)
    {
    }

    CameraParams(float fx, float fy, float cx, float cy, float bf)
        : fx(static_cast<double>(fx)),
          fy(static_cast<double>(fy)),
          cx(static_cast<double>(cx)),
          cy(static_cast<double>(cy)),
          bf(static_cast<double>(bf))
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////
// Statistics
////////////////////////////////////////////////////////////////////////////////////

/** @brief information about optimization.
 */
struct CUGO_API BatchInfo
{
    int iteration; //!< iteration number
    double chi2; //!< total chi2 (objective function value)
};


class CUGO_API BatchStatistics
{
public:
    BatchInfo& getStartStats()
    {
        assert(!stats.empty());
        return stats[0];
    }
    BatchInfo& getLastStats()
    {
        assert(!stats.empty());
        return stats.back();
    }
    BatchInfo& getStatEntry(const int idx)
    {
        assert(stats.size() < idx);
        return stats[idx];
    }
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
static constexpr Scalar ScalarCast(T v)
{
    return static_cast<Scalar>(v);
}

////////////////////////////////////////////////////////////////////////////////////
// Cuda Bundle Adjustment
////////////////////////////////////////////////////////////////////////////////////

/** @brief CUDA implementation of Bundle Adjustment.

The class implements a Bundle Adjustment algorithm with CUDA.
It optimizes camera poses and landmarks (3D points) represented by a graph.

@attention This class doesn't take responsibility for deleting pointers to vertices and edges
added in the graph.

*/
class CUGO_API CudaGraphOptimisation
{
public:
    using Ptr = UniquePtr<CudaGraphOptimisationImpl>;

    /** @brief Creates an instance of CudaGraphOptimisation.
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
    virtual ~CudaGraphOptimisation();

    virtual EdgeSetVec& getEdgeSets() = 0;

    virtual void setCameraPrams(const CameraParams& camera) = 0;

    virtual size_t nVertices(const int id) = 0;

    virtual void clearEdgeSets() = 0;
    virtual void clearVertexSets() = 0;

    virtual void setVerbose(bool status) = 0;
};

/** @brief Implementation of CudaGraphOptimisation.
 */
class CUGO_API CudaGraphOptimisationImpl : public CudaGraphOptimisation
{
public:
    /**
     * @brief constructor
     */
    CudaGraphOptimisationImpl();

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

    void setVerbose(bool status) override { verbose = status; }

    ~CudaGraphOptimisationImpl();

private:
    static inline double attenuation(double x) { return 1 - std::pow(2 * x - 1, 3); }
    static inline double clamp(double v, double lo, double hi)
    {
        return std::max(lo, std::min(v, hi));
    }

    bool verbose = false;

    VertexSetVec vertexSets;
    EdgeSetVec edgeSets;

    std::unique_ptr<BlockSolver> solver_;
    std::unique_ptr<CameraParams> camera_;

    BatchStatistics stats_;
    TimeProfile timeProfile_;
};

} // namespace cugo

#endif // !__CUDA_GRAPH_OPTIMISATION_H__
