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

#pragma once

#include "device_buffer.h"
#include "device_matrix.h"
#include "graph_optimisation_options.h"
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

/** @brief Camera parameters struct.
 */
struct CUGO_API CameraParams
{
    /// focal length x (pixel)
    double fx;
    /// focal length y (pixel)
    double fy;
    /// principal point x (pixel)
    double cx;
    /// principal point y (pixel)
    double cy;
    /// stereo baseline * fx
    double bf;

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

/** @brief information about optimization.
 */
struct CUGO_API BatchInfo
{
    int iteration; //!< iteration number
    double chi2; //!< total chi2 (objective function value)
};

/** @brief Statistical information about the overall optimisation process.
 */
class CUGO_API BatchStatistics
{
public:
    /** @brief Statisical information for the first frame of optimistaion
     * @return A BatchInfo struct with statistical info for the first frame.
     */
    BatchInfo getStartStats() const
    {
        assert(!stats.empty());
        return stats[0];
    }

    /** @brief Statisical information for the last frame of optimistaion
     * @return A BatchInfo struct with statistical info for the last frame.
     */
    BatchInfo getLastStats() const
    {
        assert(!stats.empty());
        return stats.back();
    }

    /** @brief Statisical information for the specified frame of optimistaion
     * @param idx The frame to get statistical info for.
     * @return A BatchInfo struct with statistical info for the specified frame.
     */
    BatchInfo getStatEntry(const int idx) const
    {
        assert(stats.size() < idx);
        return stats[idx];
    }

    /** @brief Adds statistical information to the container.
     * @param batchInfo The stastical information to add.
     */
    void addStat(const BatchInfo& batchInfo) { stats.push_back(std::move(batchInfo)); }

    /** @brief Returns a vector containing all stats information for the optimisation so far.
     * @return Returns a vector contains stats info.
     */
    const std::vector<BatchInfo>& get() { return stats; }

    /** @brief clears the statistical info container.
     */
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


/** @brief The CUDA graph optimisation class.
 * The class implements a graph optimisation algorithm using CUDA.
 * It optimises poses and/or landmark feature represented as a graph
 * with weight applied to each edge.
 *
 * Limitations:
 * - At present only the Levenberg-Marquardt alogrithm is supported.
 * - Camera paramters can only be set on a per edge-set basis (global)
 *
 * @attention The optimiser does not take ownership of edges or vertices so
 * the client side is required to clean up any allocated memory used.
 */
class CUGO_API CudaGraphOptimisation
{
public:
    using Ptr = UniquePtr<CudaGraphOptimisationImpl>;

    virtual ~CudaGraphOptimisation();

    /**
     * @brief Creates a new instance of CudaGraphOptimisation.
     */
    static Ptr create();

    /**
     * @brief Initializes the graph.
     */
    virtual void initialize() = 0;

    /**
     * @brief Optimizes the graph.
     * @param niterations number of iterations for Levenberg-Marquardt algorithm.
     */
    virtual void optimize(int niterations) = 0;

    /**
     * @brief Returns the batch statistics for the optimisation iterations conducted.
     */
    virtual BatchStatistics& batchStatistics() = 0;

    /**
     * @brief Returns the time profile for the optimisation iterations.
     */
    virtual const TimeProfile& timeProfile() = 0;

    /**
     * @brief Returns all the edge sets currently registered with the optimiser.
     */
    virtual EdgeSetVec& getEdgeSets() = 0;

    /** @brief Sets the camera calibration settings to use for all edges.
     * @param camera An initialised camera class. See @see CameraParams
     */
    virtual void setCameraPrams(const CameraParams& camera) = 0;

    /**
     * @brief The number of vertices addded to a specified vertex set
     * @param The id of the vertex set
     * @return The vertices count for the specified vertex set
     */
    virtual size_t nVertices(const int id) = 0;

    /**
     * @brief Clears all edge sets from the optimiser.
     */
    virtual void clearEdgeSets() = 0;

    /**
     * @brief Clears all vertex sets from the optimiser.
     */
    virtual void clearVertexSets() = 0;

    /**
     * @brief Set the Verbose object
     * @param status If true, then the optimiser will output infornmation for each iteration.
     */
    virtual void setVerbose(bool status) = 0;
};

/** @brief Implementation of CudaGraphOptimisation.
 */
class CUGO_API CudaGraphOptimisationImpl : public CudaGraphOptimisation
{
public:
    CudaGraphOptimisationImpl(GraphOptimisationOptions& options);
    CudaGraphOptimisationImpl();
    ~CudaGraphOptimisationImpl();

    /**
     * @brief adds a new graph to the optimiser with a custom edge
     * @tparam T The edge set type.
     * @param edgeSet A pointer to the edge set to add.
     * @return if the edge set is successfully added, returns true (this will always be the return
     * value)
     */
    template <typename T>
    bool addEdgeSet(T* edgeSet)
    {
        assert(edgeSet != nullptr);
        edgeSets.push_back(edgeSet);
        return true;
    }

    /**
     * @brief adds a new vertex set to the optimiser
     * @tparam T The vertex set type.
     * @param edgeSet A pointer to the vertex set to add.
     * @return if the vertex set is successfully added, returns true (this will always be the return
     * value)
     */
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

private:
    /**
     * @brief Initialise the CUDA backend. This will check that the system
     * has the required compute capability and initialise the cuda streams
     *
     */
    void initCuda();

    static inline double attenuation(double x) { return 1 - std::pow(2 * x - 1, 3); }
    static inline double clamp(double v, double lo, double hi)
    {
        return std::max(lo, std::min(v, hi));
    }

private:
    bool verbose = false;

    /// option settings for this optimiser instance
    GraphOptimisationOptions options;

    /// The vertex sets used to setup the graph for the optimisation problem
    VertexSetVec vertexSets;

    /// The edge sets used to setup the graph for the optimisation problem
    EdgeSetVec edgeSets;

    /// the block solver used for solving the graph
    std::unique_ptr<BlockSolver> solver_;

    /// The camera parameters that will be used for all edges
    std::unique_ptr<CameraParams> camera_;

    BatchStatistics stats_;
    TimeProfile timeProfile_;

    /// cuda streams
    std::array<cudaStream_t, 3> streams_;

    /// the GPU device id that will be used
    int deviceId_;

    /// Properties of the GPU that will be used.
    cudaDeviceProp deviceProp_;
};

} // namespace cugo
