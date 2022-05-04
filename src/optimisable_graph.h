
#ifndef __OPTIMISABLE_GRAPH_H__
#define __OPTIMISABLE_GRAPH_H__

#include "cuda/cuda_block_solver.h"
#include "cuda_block_solver_impl.h"
#include "cuda_bundle_adjustment.h"
#include "device_buffer.h"
#include "device_matrix.h"
#include "maths.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace cuba
{
////////////////////////////////////////////////////////////////////////////////////
// Type alias
////////////////////////////////////////////////////////////////////////////////////

template <class T>
using Set = std::unordered_set<T>;

// forward declerations
class BaseEdge;


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

    virtual void clear() = 0;
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

    void clear() override
    {
        estimates.clear();
        vertices.clear();
        activeSize = 0;
    }

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

    using Information = double;

    /** @brief Returns the vertex based on the type at index.
     */
    virtual BaseVertex* getVertex(const int index) = 0;

    /** @brief Adds a vertex of type specified by the template parameters
     */
    virtual void setVertex(BaseVertex* vertex, const int index) = 0;

    /** @brief Returns the information for this edge
     */
    virtual Information getInformation() const = 0;

    /** @brief Sets the information for this edge
     */
    virtual void setInformation(const Information& i) = 0;

    virtual bool allVerticesFixed() const = 0;

    virtual void* getMeasurement() { return nullptr; };

    /** @brief Returns the dimension of measurement.
     */
    virtual int dim() const = 0;
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
    Edge() : measurement(Measurement()), info(Information()) {}

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

    Information getInformation() const override { return info; }

    /** @brief Returns the dimension of measurement.
     */
    int dim() const override { return DIM; }

    void setMeasurement(const Measurement& m) { measurement = m; }

    void setInformation(const Information& i) override { info = i; }

protected:
    Measurement measurement;
    Information info; //!< information matrix (represented by a scalar for performance).

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

    virtual std::vector<HplBlockPos>& getHessianBlockPos() = 0;

    virtual size_t getHessianBlockPosSize() const = 0;

    virtual void init(int& edgeId, bool doSchur) = 0;

    virtual void mapDevice(int* edge2HData) = 0;

    virtual void clear() = 0;

    virtual void clearEdges() = 0;

    // device side virtual functions
    virtual void constructQuadraticForm(
        const VertexSetVec& vertexSets,
        GpuHppBlockMat& Hpp,
        GpuPx1BlockVec& bp,
        GpuHllBlockMat& Hll,
        GpuLx1BlockVec& bl,
        GpuHplBlockMat& Hpl)
    {
    }

    virtual Scalar computeError(const VertexSetVec& vertexSets, Scalar* chi) { return 0; }
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
    EdgeSet() {}
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

    const std::unordered_set<BaseEdge*>& get() override { return edges; }

    std::vector<HplBlockPos>& getHessianBlockPos() override
    {
        assert(is_initialised == true);
        return hessianBlockPos;
    }

    size_t getHessianBlockPosSize() const
    {
        assert(is_initialised == true);
        return hessianBlockPos.size();
    }

    const int dim() const override { return DIM; }

    void clearEdges() override { edges.clear(); }

protected:
    std::unordered_set<BaseEdge*> edges;

public:
    // device side

    using ErrorVec = typename std::conditional<(DIM == 1), GpuVec1d, GpuVec<VecNd<DIM>>>::type;
    using MeasurementVec = GpuVec<GpuMeasurementType>;

    void init(int& edgeId, bool doSchur) override
    {
        size_t edgeSize = edges.size();

        measurements.reserve(edgeSize);
        omegas.reserve(edgeSize);
        edge2PL.reserve(edgeSize);
        edgeFlags.reserve(edgeSize);
        hessianBlockPos.reserve(edgeSize);

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
            edge2PL.push_back(vec);

            if (doSchur && !edge->allVerticesFixed())
            {
                hessianBlockPos.push_back({vec[0], vec[1], edgeId});
            }

            omegas.push_back(ScalarCast(edge->getInformation()));
            measurements.emplace_back(*(static_cast<MeasurementType*>(edge->getMeasurement())));

            if (VertexSize == 1)
            {
                edgeFlags.push_back(
                    CudaBlockSolver::makeEdgeFlag(edge->getVertex(0)->isFixed(), false));
            }
            else
            {
                edgeFlags.push_back(CudaBlockSolver::makeEdgeFlag(
                    edge->getVertex(0)->isFixed(), edge->getVertex(1)->isFixed()));
            }
            edgeId++;
        }
        is_initialised = true;
    }

    void mapDevice(int* edge2HData) override
    {
        assert(is_initialised == true);
        size_t edgeSize = edges.size();
        d_measurements.assign(edgeSize, measurements.data());
        d_errors.resize(edgeSize);
        d_omegas.assign(edgeSize, omegas.data());
        d_Xcs.resize(edgeSize);
        d_edgeFlags.assign(edgeSize, edgeFlags.data());
        d_edge2PL.assign(edgeSize, edge2PL.data());
        if (edge2HData)
        {
            d_edge2Hpl.map(edgeSize, edge2HData);
        }
    }

    void clear() override
    {
        hessianBlockPos.clear();
        omegas.clear();
        edge2PL.clear();
        edgeFlags.clear();
        measurements.clear();
    }

protected:
    bool is_initialised = false;

    // cpu
    std::vector<Scalar> omegas;
    std::vector<VIndex> edge2PL;
    std::vector<uint8_t> edgeFlags;
    std::vector<MeasurementType> measurements;
    std::vector<HplBlockPos> hessianBlockPos;

    // device
    GpuVec3d d_Xcs;
    GpuVec1d d_omegas;
    MeasurementVec d_measurements;
    ErrorVec d_errors;
    GpuVec2i d_edge2PL;
    GpuVec1b d_edgeFlags;
    GpuVec1i d_edge2Hpl;
};

////////////////////////////////////////////////////////////////////////////////////
// Camera parameters
////////////////////////////////////////////////////////////////////////////////////

/** @brief Camera parameters struct.
 */
struct CameraParams
{
    double fx; //!< focal length x (pixel)
    double fy; //!< focal length y (pixel)
    double cx; //!< principal point x (pixel)
    double cy; //!< principal point y (pixel)
    double bf; //!< stereo baseline times fx

    /** @brief The constructor.
     */
    CameraParams() : fx(0), fy(0), cx(0), cy(0), bf(0) {}
};
;

#include "optimisable_graph.hpp"

} // namespace cuba

#endif // __OPTIMISABLE_GRAPH_H__
