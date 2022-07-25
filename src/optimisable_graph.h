
#ifndef __OPTIMISABLE_GRAPH_H__
#define __OPTIMISABLE_GRAPH_H__

#include "arena.h"
#include "async_vector.h"
#include "block_solver.h"
#include "cuda/cuda_block_solver.h"
#include "cuda_graph_optimisation.h"
#include "device_buffer.h"
#include "device_matrix.h"
#include "maths.h"
#include "sparse_block_matrix.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace cugo
{

template <class T>
using Set = std::unordered_set<T>;

// forward declerations
class BaseEdge;
class BaseRobustKernel;


////////////////////////////////////////////////////////////////////////////////////
// Vertex
////////////////////////////////////////////////////////////////////////////////////

class BaseVertex
{
public:
    BaseVertex() {};
    virtual ~BaseVertex() {}

    virtual int getId() const = 0;

    virtual void setId(const int id) = 0;

    virtual Set<BaseEdge*>& getEdges() = 0;

    virtual void addEdge(BaseEdge* edge) = 0;

    virtual void removeEdge(BaseEdge* edge) = 0;

    virtual void setFixed(bool status) = 0;

    virtual bool isFixed() const = 0;

    virtual int getIndex() const = 0;

    virtual void setIndex(const int idx) = 0;

    virtual bool isMarginilised() const = 0;

    virtual void clearEdges() = 0;
};

template <typename T, bool Marginilised>
class Vertex : public BaseVertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using EstimateType = T;
    const bool marginilised = Marginilised;

    Vertex() {}
    virtual ~Vertex() {}

    Vertex(int id, const EstimateType& est, bool fixed = false)
        : estimate(est), fixed(fixed), id(id), idx(-1)
    {
    }

    Vertex(int id, bool fixed = false) : fixed(fixed), id(id), idx(-1) {}

    EstimateType& getEstimate() { return estimate; }

    void setEstimate(const EstimateType& est) { estimate = est; }

    Set<BaseEdge*>& getEdges() override { return edges; }

    void addEdge(BaseEdge* edge) override
    {
        assert(edge != nullptr);
        edges.insert(edge);
    }

    void removeEdge(BaseEdge* edge) override
    {
        assert(edge != nullptr);
        edges.erase(edge);
    }

    void setFixed(bool status) override { fixed = status; }

    bool isFixed() const override { return fixed; }

    void setId(const int id) override { this->id = id; }

    int getId() const override { return id; }

    int getIndex() const override { return idx; }

    void setIndex(const int idx) override { this->idx = idx; }

    bool isMarginilised() const override { return marginilised; }

    void clearEdges() override { edges.clear(); }

protected:
    EstimateType estimate;
    bool fixed; //!< if true, the state variables are fixed during optimization.
    int id; //!< ID of the vertex.
    int idx; //!< ID of the vertex (internally used).
    Set<BaseEdge*> edges; //!< connected edges.
};

using PoseVertex = Vertex<maths::Se3D, false>;
using LandmarkVertex = Vertex<maths::Vec3d, true>;

////////////////////////////////////////////////////////////////////////////////////
// Vertex set
////////////////////////////////////////////////////////////////////////////////////

class BaseVertexSet
{
public:
    BaseVertexSet() {};
    virtual ~BaseVertexSet() {}

    virtual bool removeVertex(BaseVertex* v, BaseEdgeSet* edgeSet, BaseVertexSet* vertexSet) = 0;

    virtual size_t size() const = 0;

    virtual size_t estimateDataSize() const = 0;

    virtual size_t getDeviceEstimateSize() = 0;

    virtual bool isMarginilised() const = 0;

    virtual int getActiveSize() const = 0;

    virtual void clearEstimates() = 0;

    virtual void clearVertices() = 0;
};

template <typename T, typename E, typename D>
class VertexSet : public BaseVertexSet
{
public:
    using VertexType = T;
    using EstimateType = E;
    using DeviceType = D;
    using DeviceVecType = GpuVec<DeviceType>;

    VertexSet(bool marg) : marginilised(marg) {}
    virtual ~VertexSet() {}

    void addVertex(T* vertex)
    {
        assert(vertex != nullptr);
        vertexMap.emplace(vertex->getId(), std::move(vertex));
    }

    T* getVertex(const int id) const;

    bool removeVertex(BaseVertex* v, BaseEdgeSet* edgeSet, BaseVertexSet* vertexSet) override;

    size_t size() const override { return vertexMap.size(); }

    bool isMarginilised() const override { return marginilised; }

protected:
    std::map<int, VertexType*> vertexMap; //!< connected vertices.
    bool marginilised; //!< landmark vertices are marginilised during optimistaion (set to false for
                       //!< pose)

public:
    // device functions
    void mapEstimateData(Scalar* d_dataPtr);

    void finalise();

    size_t estimateDataSize() const override { return estimates.size(); }

    void* getDeviceEstimateData() { return static_cast<void*>(d_estimate.data()); }
    size_t getDeviceEstimateSize() override { return d_estimate.size(); }

    std::vector<T*>& get() { return vertices; }

    DeviceVecType& getDeviceEstimates() { return d_estimate; }
    std::vector<DeviceType>& getEstimates() { return estimates; }

    int getActiveSize() const override { return activeSize; }

    void clearEstimates() override
    {
        estimates.clear();
        vertices.clear();
        activeSize = 0;
    }

    void clearVertices() override { vertexMap.clear(); }

private:
    // gpu hosted estimate data vec
    DeviceVecType d_estimate;

    // cpu-gpu estimate data
    std::vector<DeviceType> estimates;

    std::vector<VertexType*> vertices;

    int activeSize = 0;
};

using PoseVertexSet = VertexSet<PoseVertex, maths::Se3D, Se3D>;
using LandmarkVertexSet = VertexSet<LandmarkVertex, maths::Vec3d, Vec3d>;


////////////////////////////////////////////////////////////////////////////////////
// Edge
////////////////////////////////////////////////////////////////////////////////////

/** @brief Base edge struct.
 */
class BaseEdge
{
public:
    BaseEdge() = default;
    virtual ~BaseEdge() {}

    /** @brief Returns the vertex based on the type at index.
     */
    virtual BaseVertex* getVertex(const int index) = 0;

    /** @brief Adds a vertex of type specified by the template parameters
     */
    virtual void setVertex(BaseVertex* vertex, const int index) = 0;

    virtual bool allVerticesFixed() const = 0;

    virtual void* getMeasurement() { return nullptr; };

    /** @brief Returns the dimension of measurement.
     */
    virtual int dim() const = 0;

#ifdef USE_PER_EDGE_INFORMATION
    using Information = double;
    /** @brief Sets the information for this edge. This should only be used
     * if it's known that edges will have differing information values
     * as this does have a performance cost due to the extra bandwidth required
     * to upload and iterate over the values.
     */
    virtual void setInformation(const Information info) = 0;

    /** @brief Returns the global information for this edge set.
     */
    virtual Information getInformation() = 0;
#endif
};

/** @brief Edge with N-dimensional measurement.
@tparam DIM dimension of the measurement vector.
*/
template <int DIM, typename E, typename... VertexTypes>
class Edge : public BaseEdge
{
public:
    using Measurement = E;

    static constexpr auto VertexSize = sizeof...(VertexTypes);

    template <int N, typename... Types>
    using VertexNth = typename std::tuple_element<N, std::tuple<Types...>>::type;

    template <int N>
    using VertexNthType = VertexNth<N, VertexTypes...>;

    template <int N>
    const VertexNthType<N>* getVertexN() const
    {
        return static_cast<const VertexNthType<N>*>(vertices[N]);
    }
    template <int N>
    VertexNthType<N>* getVertexN()
    {
        return static_cast<VertexNthType<N>*>(vertices[N]);
    }

    /** @brief The constructor.
     */
    Edge() : measurement(Measurement()) {}

    /** @brief the destructor.
     */
    virtual ~Edge() {}

    BaseVertex* getVertex(const int index) override { return vertices[index]; }

    void setVertex(BaseVertex* vertex, const int index) override { vertices[index] = vertex; }

    template <std::size_t... Ints>
    bool allVerticesFixedNs(std::index_sequence<Ints...>) const
    {
        bool fixed[] = {getVertexN<Ints>()->isFixed()...};
        return std::all_of(std::begin(fixed), std::end(fixed), [](bool value) { return value; });
    }

    bool allVerticesFixed() const override
    {
        return allVerticesFixedNs(std::make_index_sequence<VertexSize>());
    }

    /** @brief Returns the dimension of measurement.
     */
    int dim() const override { return DIM; }

    void setMeasurement(const Measurement& m) { measurement = m; }

#ifdef USE_PER_EDGE_INFORMATION
    void setInformation(const Information info) override { info_ = info; }

    Information getInformation() override { return info_; }
#endif

protected:
    Measurement measurement;

#ifdef USE_PER_EDGE_INFORMATION
    Information info_; //!< information matrix (represented by a scalar for performance).
#endif

    BaseVertex* vertices[VertexSize];
};


////////////////////////////////////////////////////////////////////////////////////
// EdgeSet
////////////////////////////////////////////////////////////////////////////////////

class BaseEdgeSet
{
public:
    virtual void addEdge(BaseEdge* edge) = 0;

    virtual void removeEdge(BaseEdge* edge) = 0;

    virtual size_t nedges() const = 0;

    virtual const std::unordered_set<BaseEdge*>& get() = 0;

    virtual const int dim() const = 0;

    virtual void* getHessianBlockPos() = 0;

    virtual size_t getHessianBlockPosSize() const = 0;

    virtual void init(Arena& hBlockPosArena, int& edgeId, cudaStream_t stream, bool doSchur) = 0;

    virtual void mapDevice(int* edge2HData, cudaStream_t stream) = 0;

    virtual void clearDevice() = 0;

    virtual void clearEdges() = 0;

    virtual std::vector<int>& outliers() = 0;

#ifndef USE_PER_EDGE_INFORMATION
    // forced to double for now - should be derided from Scalar type.
    using Information = double;

    /** @brief Sets the global information for this edge set which will applied to
     * all edges. This is for performance purposes and the fact that all optimisations
     * appear to have the same information value.
     */
    virtual void setInformation(const Information info) = 0;

    /** @brief Returns the global information for this edge set.
     */
    virtual Information getInformation() = 0;
#endif

    // device side virtual functions
    virtual void constructQuadraticForm(
        const VertexSetVec& vertexSets,
        GpuPxPBlockVec& Hpp,
        GpuPx1BlockVec& bp,
        GpuLxLBlockVec& Hll,
        GpuLx1BlockVec& bl,
        GpuHplBlockMat& Hpl,
        cudaStream_t stream)
    {
    }

    virtual Scalar computeError(const VertexSetVec& vertexSets, Scalar* chi, cudaStream_t stream)
    {
        return 0;
    }

    virtual void setRobustKernel(BaseRobustKernel* kernel) = 0;
    virtual BaseRobustKernel* robustKernel() = 0;
};

/** @brief groups together a set of edges of the same type.
 */
template <int DIM, typename E, typename F, typename... VertexTypes>
class EdgeSet : public BaseEdgeSet
{
public:
    using MeasurementType = E;
    using GpuMeasurementType = F;

    static constexpr auto VertexSize = sizeof...(VertexTypes);

    // Note: even though only this edgeset may denote a single vertex type,
    // we always state two integers as this is what is expected on the GPU side
    // If only one vertex is stated, the second element will be ignored.
    using VIndex = std::array<int, 2>;

    // cpu side
    EdgeSet() : kernel(nullptr), outlierThreshold(0.0) {}
    virtual ~EdgeSet() {}

    // vitual functions
    void addEdge(BaseEdge* edge) override
    {
        for (int i = 0; i < VertexSize; ++i)
        {
            edge->getVertex(i)->addEdge(edge);
        }
        edges.insert(edge);
    }

    void removeEdge(BaseEdge* edge) override
    {
        for (int i = 0; i < VertexSize; ++i)
        {
            BaseVertex* vertex = edge->getVertex(i);
            if (vertex->getEdges().count(edge))
            {
                vertex->removeEdge(edge);
            }
        }
        if (edges.count(edge))
        {
            edges.erase(edge);
        }
    }

    size_t nedges() const override { return edges.size(); }

    const std::unordered_set<BaseEdge*>& get() { return edges; }

#ifndef USE_PER_EDGE_INFORMATION
    void setInformation(const Information info) override { info_ = info; };

    Information getInformation() override { return info_; }
#endif

    void* getHessianBlockPos() override { return hessianBlockPos->data(); }

    size_t getHessianBlockPosSize() const override { return hessianBlockPos->size(); }

    const int dim() const override { return DIM; }

    void setRobustKernel(BaseRobustKernel* kernel) override { this->kernel = kernel; }
    BaseRobustKernel* robustKernel() override { return kernel; }

    void setOutlierThreshold(const Scalar errorThreshold)
    {
        this->outlierThreshold = errorThreshold;
    }

    std::vector<int>& outliers() override
    {
        assert(
            outlierThreshold > 0.0 &&
            "No error threshold set for this edgeSet, thus no outliers will have been calcuated "
            "during graph optimisation.");
        edgeLevels.resize(edges.size());
        d_outliers.copyTo(edgeLevels.data());
        return edgeLevels;
    }

    void clearEdges() override { edges.clear(); }

protected:
    std::unordered_set<BaseEdge*> edges;
    BaseRobustKernel* kernel;
    Scalar outlierThreshold;
    std::vector<int> edgeLevels;
    size_t totalBufferSize_ = 0;
#ifndef USE_PER_EDGE_INFORMATION
    Information info_; //!< information matrix (represented by a scalar for performance).
#endif

public:
    // device side

    using ErrorVec = typename std::conditional<(DIM == 1), GpuVec1d, GpuVec<VecNd<DIM>>>::type;
    using MeasurementVec = GpuVec<GpuMeasurementType>;

    void init(Arena& hBlockPosArena, int& edgeId, cudaStream_t stream, bool doSchur) override
    {
        size_t edgeSize = edges.size();
        totalBufferSize_ = sizeof(MeasurementType) * edgeSize + sizeof(VIndex) * edgeSize +
            sizeof(uint8_t) * edgeSize;
#ifdef USE_PER_EDGE_INFORMATION
        totalBufferSize_ += sizeof(Scalar) * edgeSize;
#else
        totalBufferSize_ += sizeof(Scalar);
#endif

        // allocate more buffers than needed to reduce the need
        // for resizing.
        arena.resize(totalBufferSize_ * 2);
        measurements = arena.allocate<MeasurementType>(edgeSize);
        edge2PL = arena.allocate<VIndex>(edgeSize);
        edgeFlags = arena.allocate<uint8_t>(edgeSize);
#ifdef USE_PER_EDGE_INFORMATION
        omega = arena.allocate<Scalar>(edgeSize);
#endif

        // all heassian block positions are also
        if (doSchur)
        {
            hessianBlockPos = hBlockPosArena.allocate<HplBlockPos>(edgeSize);
        }

        for (BaseEdge* edge : edges)
        {
            VIndex vec;
            for (int i = 0; i < VertexSize; ++i)
            {
                BaseVertex* vertex = edge->getVertex(i);
                // non-marginilised (pose) indices are first
                if (!vertex->isMarginilised())
                {
                    vec[0] = vertex->getIndex();
                    assert(vec[0] != -1);
                }
                else
                {
                    vec[1] = vertex->getIndex();
                    assert(vec[1] != -1);
                }
            }
            edge2PL->push_back(vec);

            if (doSchur && !edge->allVerticesFixed())
            {
                hessianBlockPos->push_back({vec[0], vec[1], edgeId});
            }

#ifdef USE_PER_EDGE_INFORMATION
            omega->push_back(ScalarCast(edge->getInformation()));
#endif
            measurements->push_back(*(static_cast<MeasurementType*>(edge->getMeasurement())));

            if (VertexSize == 1)
            {
                edgeFlags->push_back(
                    BlockSolver::makeEdgeFlag(edge->getVertex(0)->isFixed(), false));
            }
            else
            {
                edgeFlags->push_back(BlockSolver::makeEdgeFlag(
                    edge->getVertex(0)->isFixed(), edge->getVertex(1)->isFixed()));
            }
            edgeId++;
        }
    }

    void mapDevice(int* edge2HData, cudaStream_t stream) override
    {
        size_t edgeSize = edges.size();

        // buffers filled by the gpu kernels.
        d_errors.resize(edgeSize);
        d_Xcs.resize(edgeSize);

        if (outlierThreshold > 0.0)
        {
            d_outlierThreshold.assign(1, &outlierThreshold);
            d_outliers.resize(edgeSize);
        }
        if (edge2HData)
        {
            d_edge2Hpl.map(edgeSize, edge2HData);
        }

        // The main mega buffer which contains all of the data used
        // in optimising the graph - transferring one large buffer async
        // is far more optimal than transferring multiple smaller buffers
        d_dataBuffer.assignAsync(totalBufferSize_, arena.data(), stream);

        d_edgeFlags.offset(d_dataBuffer, edgeSize, edgeFlags->bufferOffset());
        d_edge2PL.offset(d_dataBuffer, edgeSize, edge2PL->bufferOffset());
        d_measurements.offset(d_dataBuffer, edgeSize, measurements->bufferOffset());
#ifdef USE_PER_EDGE_INFORMATION
        d_omega.offset(d_dataBuffer, edgeSize, omega->bufferOffset());
#else
        // TODO: ideally this would be allocated on the memory pool rather than
        // a seperate upload but for some reason using the arena gives memory
        // alignment issues in cuda.
        d_omega.assignAsync(1, &info_);
#endif
    }

    void clearDevice() override { arena.clear(); }

protected:
    // cpu - using pinned memory for async access
    Arena arena;
    std::unique_ptr<ArenaPtr<Scalar>> omega;
    std::unique_ptr<ArenaPtr<VIndex>> edge2PL;
    std::unique_ptr<ArenaPtr<uint8_t>> edgeFlags;
    std::unique_ptr<ArenaPtr<MeasurementType>> measurements;
    std::unique_ptr<ArenaPtr<HplBlockPos>> hessianBlockPos;

    // device
    GpuVec<uint8_t> d_dataBuffer;
    GpuVec3d d_Xcs;
    GpuVec<Scalar> d_omega;
    MeasurementVec d_measurements;
    ErrorVec d_errors;
    GpuVec2i d_edge2PL;
    GpuVec1b d_edgeFlags;
    GpuVec1i d_edge2Hpl;
    DeviceBuffer<Scalar> d_outlierThreshold;
    GpuVec1i d_outliers;
};


#include "optimisable_graph.hpp"

} // namespace cugo

#endif // __OPTIMISABLE_GRAPH_H__
