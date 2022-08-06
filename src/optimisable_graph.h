
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

    /**
     * @brief Get the id of the vertex.
     * @return The id of the vertex.
     */
    virtual int getId() const noexcept = 0;

    /**
     * @brief Set the id of the vertex
     * @param id The id number the vertex will use.
     */
    virtual void setId(const int id) noexcept = 0;

    /**
     * @brief Returns all the edges associated with the vertex
     * @return A container with the edges added to this vertex
     */
    virtual Set<BaseEdge*>& getEdges() noexcept = 0;

    /**
     * @brief Adds an edge to the vertex
     * @param edge A pointer to the edge to add.
     */
    virtual void addEdge(BaseEdge* edge) = 0;

    /**
     * @brief Removes an edge from the vertex
     * @param edge A pointer to the edge to remove.
     */
    virtual void removeEdge(BaseEdge* edge) = 0;

    /**
     * @brief States whether this vertex will have a hard constraint applied and will
     * stay at its initial values.
     * @param status States whether this edge is fixed (true) or not (false)
     */
    virtual void setFixed(bool status) noexcept = 0;

    /**
     * @brief Returns whether this vertex is fixed or not.
     * @return If fixed, returns true. Otherwise, false.
     */
    virtual bool isFixed() const noexcept = 0;

    virtual int getIndex() const noexcept = 0;

    virtual void setIndex(const int idx) noexcept = 0;

    virtual bool isMarginilised() const noexcept = 0;

    /**
     * @brief Clears all edges from this vertex.
     */
    virtual void clearEdges() noexcept = 0;
};

/**
 * @brief A vertex used in the graph optimisation problem.
 *
 * @tparam T The type of this vertex.
 * @tparam Marginilised Whether this is a marginilised vertex as per Shcur complement calculations
 * for Bundle Adjustemt. Pose vertices are usually not marginilised (false), whereas Landmark
 * vertices are marginilised.
 */
template <typename T, bool Marginilised>
class Vertex : public BaseVertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using EstimateType = T;
    const bool marginilised = Marginilised;

    Vertex() {}
    Vertex(int id, const EstimateType& est, bool fixed = false)
        : estimate(est), fixed(fixed), id(id), idx(-1)
    {
    }
    Vertex(int id, bool fixed = false) : fixed(fixed), id(id), idx(-1) {}

    virtual ~Vertex() {}

    EstimateType& getEstimate() noexcept;
    void setEstimate(const EstimateType& est) noexcept;
    Set<BaseEdge*>& getEdges() noexcept override;
    void addEdge(BaseEdge* edge) override;
    void removeEdge(BaseEdge* edge) override;
    void setFixed(bool status) noexcept override;
    bool isFixed() const noexcept override;
    void setId(const int id) noexcept override;
    int getId() const noexcept override;
    int getIndex() const noexcept override;
    void setIndex(const int idx) noexcept override;
    bool isMarginilised() const noexcept override;
    void clearEdges() noexcept override;

protected:
    /// The estimate for this vertex.
    EstimateType estimate;
    /// if true, the state variables are fixed during optimization.
    bool fixed;
    /// ID of the vertex.
    int id;
    ///  ID of the vertex (internally used).
    int idx;
    ///  connected edges.
    Set<BaseEdge*> edges;
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

    /**
     * @brief Remove a vertex from the set.
     * @param v Pointer to vertex to remove
     * @param edgeSet The edge set associated with the vertex to remove - will  be removed from here
     * to.
     * @return If successfully removed, returns true.
     */
    virtual bool removeVertex(BaseVertex* v, BaseEdgeSet* edgeSet) = 0;

    /**
     * @brief The number of vertices in this set
     * @return Returns the vertices count.
     */
    virtual size_t size() const noexcept = 0;

    /**
     * @brief The number of estimates associated with this set on the host.
     * @return The host estimates count.
     */
    virtual size_t estimateDataSize() const noexcept = 0;

    /**
     * @brief The number of estimates associated with this set on the device.
     * @return The device estimates count.
     */
    virtual size_t getDeviceEstimateSize() noexcept = 0;

    /**
     * @brief Retuens the marginilised state of this set.
     * @return If the set is marginilised, returns true.
     */
    virtual bool isMarginilised() const noexcept = 0;

    /**
     * @brief Returns the number of active (non-fixed) vertices in the set.
     * @return The active vertices count.
     */
    virtual int getActiveSize() const noexcept = 0;

    /**
     * @brief Clears all of the host estimates from the set.
     */
    virtual void clearEstimates() noexcept = 0;

    /**
     * @brief Clears all of the vertices from the set.
     */
    virtual void clearVertices() noexcept = 0;
};

/**
 * @brief A collection of vertices and their data
 *
 * @tparam T The type for the vertex
 * @tparam E The type for the estimate (host)
 * @tparam D The type for the device estimate. This should correlate with type @see E
 */
template <typename T, typename E, typename D>
class VertexSet : public BaseVertexSet
{
public:
    using VertexType = T;
    using EstimateType = E;
    using DeviceType = D;
    using DeviceVecType = GpuVec<DeviceType>;

    // static_assert(EstimateType == T::EstimateType);

    VertexSet(bool marg) : marginilised(marg) {}
    virtual ~VertexSet() {}

    // non-virtual functions
    /**
     * @brief Adds a vertex to the set
     * @param vertex A pointer to the vertex to add.
     */
    void addVertex(T* vertex);

    /**
     * @brief Get the specified vertex from the set.
     * @param id The id of the vertex to return
     * @return The vertex assocaited with the specified id.
     */
    T* getVertex(const int id) const;

    // virtual functions
    bool removeVertex(BaseVertex* v, BaseEdgeSet* edgeSet) override;
    size_t size() const noexcept override;
    bool isMarginilised() const noexcept override;

protected:
    std::map<int, VertexType*> vertexMap; //!< connected vertices.
    bool marginilised; //!< landmark vertices are marginilised during optimistaion (set to false for
                       //!< pose)

public:
    // device functions
    // non-virtual functions
    /**
     * @brief Maps the specified data onto the device allocated space
     * @param d_dataPtr A pointer to the data that will be uploaded
     */
    void mapEstimateData(Scalar* d_dataPtr);

    /**
     * @brief Copy the estimate from the device to the host estimate container.
     * Used to update the host once optimisation has been conducted.
     */
    void finalise();

    /**
     * @brief Get the allocated device memory pointer
     * @return Returns the estimates allocated device memory address as a void type
     */
    void* getDeviceEstimateData() noexcept;

    /**
     * @brief Get all of the vertices associated with this set.
     * @return A vector of vertices.
     */
    std::vector<T*>& get() noexcept;

    /**
     * @brief Returns the estimates as a device buffer.
     * @return A @see DeviceVecType buffer conatining estimates
     */
    DeviceVecType& getDeviceEstimates() noexcept;

    /**
     * @brief Get the estimates associated with this vertex set (host)
     * @return A vector of estimates
     */
    std::vector<DeviceType>& getEstimates() noexcept;

    // virtual functions
    size_t estimateDataSize() const noexcept override;
    size_t getDeviceEstimateSize() noexcept override;
    int getActiveSize() const noexcept override;
    void clearEstimates() noexcept override;
    void clearVertices() noexcept override;

private:
    /// gpu hosted estimate data vec
    DeviceVecType d_estimate;

    /// cpu-gpu estimate data
    std::vector<DeviceType> estimates;

    /// the vertices associated with this set
    std::vector<VertexType*> vertices;

    /// the number of active vertices
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
    using Information = Scalar;

    BaseEdge() = default;
    virtual ~BaseEdge() {}

    /**
     * @brief Returns the vertex based on the type at index.
     */
    virtual BaseVertex* getVertex(const int index) = 0;

    /**
     * @brief Adds a vertex of type specified by the template parameters
     */
    virtual void setVertex(BaseVertex* vertex, const int index) = 0;

    /**
     * @brief Check whether all vertex types are fixed.
     * @return If all vertex types are fixed, returns true.
     */
    virtual bool allVerticesFixed() const noexcept = 0;

    /**
     * @brief Get the measurement associated with the edge.
     * @return Returns the measurment as a void pointer.
     */
    virtual void* getMeasurement() noexcept { return nullptr; };

    /**
     * @brief Returns the dimension of measurement.
     * @return The measurements dimensions
     */
    virtual int dim() const noexcept = 0;

    /**
     * @brief Sets the weight to add for this edge. Note:
     * @param info The weight value to set for this edge
     */
    virtual void setInformation(const Information info) noexcept = 0;

    /**
     * @brief Get the weight associated with this edge
     * @return The weight as type @see Information
     */
    virtual Information getInformation() noexcept = 0;
};

/**
 * @brief Edge with N-dimensional measurement.
 * @tparam DIM dimension of the measurement vector.
 * @tparam E The measurement type for this edge
 * @tparam VertexTypes A varadic template of vertex types associated with this edge.
 */
template <int DIM, typename E, typename... VertexTypes>
class Edge : public BaseEdge
{
public:
    using Measurement = E;

    /// The number of vertex types for this edge
    static constexpr auto VertexSize = sizeof...(VertexTypes);

    /// The Nth Vertex type in the tuple
    template <int N, typename... Types>
    using VertexNth = typename std::tuple_element<N, std::tuple<Types...>>::type;

    template <int N>
    using VertexNthType = VertexNth<N, VertexTypes...>;

    /// get a vertex from the template list as a constant
    template <int N>
    const VertexNthType<N>* getVertexN() const
    {
        return static_cast<const VertexNthType<N>*>(vertices[N]);
    }

    /// get a vertex from the template list
    template <int N>
    VertexNthType<N>* getVertexN()
    {
        return static_cast<VertexNthType<N>*>(vertices[N]);
    }

    Edge() : measurement(Measurement()) {}
    virtual ~Edge() {}

    // virtual functions
    BaseVertex* getVertex(const int index) override;
    void setVertex(BaseVertex* vertex, const int index) override;
    bool allVerticesFixed() const noexcept override;
    int dim() const noexcept override;
    void setInformation(const Information info) noexcept override;
    Information getInformation() noexcept override;

    // non-virtual functions
    template <std::size_t... Ints>
    bool allVerticesFixedNs(std::index_sequence<Ints...>) const
    {
        bool fixed[] = {getVertexN<Ints>()->isFixed()...};
        return std::all_of(std::begin(fixed), std::end(fixed), [](bool value) { return value; });
    }

    /**
     * @brief Set the measurement for the edge.
     * @param The measurement
     */
    void setMeasurement(const Measurement& m) noexcept;

protected:
    /// The measurement for the edge
    Measurement measurement;
    /// information matrix (represented by a scalar for performance).
    Information info_; //!<
    /// The vertex types for the edge
    BaseVertex* vertices[VertexSize];
};


////////////////////////////////////////////////////////////////////////////////////
// EdgeSet
////////////////////////////////////////////////////////////////////////////////////

class BaseEdgeSet
{
public:
    using Information = Scalar;

    /**
     * @brief Add an edge to the set. Type is determined by @see EdgeSet
     * @param edge The edge to add.
     */
    virtual void addEdge(BaseEdge* edge) = 0;

    /**
     * @brief Remove an edge from the set.
     * @param The edge to remove.
     */
    virtual void removeEdge(BaseEdge* edge) = 0;

    /**
     * @brief The number of edges associated with this set.
     * @return The edge count.
     */
    virtual size_t nedges() const noexcept = 0;

    /**
     * @brief Get a container of edges associated with this set.
     * @return A vector of edges.
     */
    virtual const std::unordered_set<BaseEdge*>& get() noexcept = 0;

    /**
     * @brief Dimension of the measurement vector associated with the edges.
     * This must be identical to the @see Edge DIM
     * @return The dimensions of the measurement vector.
     */
    virtual const int dim() const noexcept = 0;

    /**
     * @brief Get the Hessian Block position data created by the @see init function.
     * @return A pointer to the hessian block position data.
     */
    virtual void* getHessianBlockPos() noexcept = 0;

    /**
     * @brief The number of elements in the hessian block position data field.
     * @return The block position count.
     */
    virtual size_t getHessianBlockPosSize() const noexcept = 0;

    /**
     * @brief Initialise the edge vertex set.
     * @param hBlockPosArena If using the schur complement, this defines the memory allocation
     * poolfor the block positions.
     * @param edgeIdOffset The offset that the edge ids will begin from.
     * @param stream A CUDA stream object.
     * @param doSchur States whether this optimiser will conduct Schur complement calculations
     * @param options A @see GraphOptimisationOptions object
     */
    virtual void init(
        Arena& hBlockPosArena,
        const int edgeIdOffset,
        cudaStream_t stream,
        bool doSchur,
        const GraphOptimisationOptions& options) = 0;

    /**
     * @brief Maps the data derived from the @see init call to the device.
     * @param edge2HData A pointer to an edge set hessian data used when Schur complemenet is
     * active.
     * @param stream A CUDA stream object.
     * @param options A @see GraphOptimisationOptions object
     */
    virtual void
    mapDevice(int* edge2HData, cudaStream_t stream, const GraphOptimisationOptions& options) = 0;

    /**
     * @brief Clear the device side containers in this set. Note: This does not deallocate device
     * memory.
     */
    virtual void clearDevice() noexcept = 0;

    /**
     * @brief Clear the edges from the set.
     */
    virtual void clearEdges() noexcept = 0;

    /**
     * @brief Return outliers that are calculated on the device. This is only relevant when @see
     * outlierThreshold is greater than 0.0
     * @return A vector of outliers.
     */
    virtual std::vector<int>& outliers() = 0;

    /**
     * @brief Set the Robust Kernel associated with this set (applied to all edges)
     * @param kernel A initialised Robust Kernel class @see BaseRobustKernel
     */
    virtual void setRobustKernel(BaseRobustKernel* kernel) noexcept = 0;

    /**
     * @brief Get the Robust Kernel associated with this set.
     * @return A pointer to the RobustKernel class associated with this set. If not set, will be
     * nullptr.
     */
    virtual BaseRobustKernel* robustKernel() noexcept = 0;

    /**
     * @brief Sets the information for this edge set.
     */
    virtual void setInformation(const Information info) noexcept = 0;

    /**
     * @brief Returns the global information for this edge set.
     */
    virtual Information getInformation() noexcept = 0;

    // device side virtual functions
    /**
     * @brief Constructs the quadratic equation for this set.
     *
     * @param vertexSets A vector of VertexSets that constitute the "nodes" to be used in the
     * optimisation algorithm
     * @param Hpp An initialised pose hessian matrix (device). This will be written to by the call.
     * @param bp
     * @param Hll An initialised landmark hessian matrix (device). This will be written to by the
     * call. Not used if no landmark (marginilised) vertex is part of the optimisation.
     * @param bl
     * @param Hpl An initialised pose-landmark hessian matrix as used in Schur complement
     * calculations (device). This will be written to by the call. Not used if no landmark
     * (marginilised) vertex is part of the optimisation.
     * @param stream A CUDA stream object
     */
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

    /**
     * @brief Calculate the errors based on the optimisation inputs.
     *
     * @param vertexSets A vector of VertexSets that constitute the "nodes" to be used in the
     * optimisation algorithm
     * @param chi Device memory allocation where the chi value will be stored on the device side.
     * @param stream A CUDA stream object
     * @return Scalar The calculated chi2 value
     */
    virtual Scalar computeError(const VertexSetVec& vertexSets, Scalar* chi, cudaStream_t stream)
    {
        return 0;
    }
};

/**
 * @brief Group together a set of edges of the same type.
 * @tparam DIM dimension of the measurement vector.
 * @tparam E The measurement type for this edge (host)
 * @tparam F The measurement type for this edge (device). TODO: Merge the E and F template paramters
 * into one.
 * @tparam VertexTypes A varadic template of vertex types associated with this edge.
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

    // host side
    EdgeSet() : kernel(nullptr), outlierThreshold(0.0), info_(0.0), totalBufferSize_(0) {}
    virtual ~EdgeSet() {}

    // vitual functions
    void addEdge(BaseEdge* edge) override;
    void removeEdge(BaseEdge* edge) override;
    size_t nedges() const noexcept override;
    const std::unordered_set<BaseEdge*>& get() noexcept;
    void* getHessianBlockPos() noexcept override;
    size_t getHessianBlockPosSize() const noexcept override;
    const int dim() const noexcept override;
    void setRobustKernel(BaseRobustKernel* kernel) noexcept override;
    BaseRobustKernel* robustKernel() noexcept override;
    std::vector<int>& outliers() override;
    void clearEdges() noexcept override;
    void setInformation(const Information info) noexcept override;
    Information getInformation() noexcept override;

    // non-virtual functions
    /**
     * @brief Set the threshold in which a error value is considered an outlier
     * @param errorThreshold The outlier threshold value
     */
    void setOutlierThreshold(const Scalar errorThreshold) noexcept;

protected:
    std::unordered_set<BaseEdge*> edges;
    BaseRobustKernel* kernel;
    Scalar outlierThreshold;
    std::vector<int> edgeLevels;
    size_t totalBufferSize_;
    Information info_;

public:
    // device side
    using ErrorVec = typename std::conditional<(DIM == 1), GpuVec1d, GpuVec<VecNd<DIM>>>::type;
    using MeasurementVec = GpuVec<GpuMeasurementType>;

    void init(
        Arena& hBlockPosArena,
        const int edgeIdOffset,
        cudaStream_t stream,
        bool doSchur,
        const GraphOptimisationOptions& options) override;

    void mapDevice(
        int* edge2HData, cudaStream_t stream, const GraphOptimisationOptions& options) override;

    void clearDevice() noexcept override;

protected:
    /// A memory pool for allocating a chunk of memory for all edge set data. Uses pinned memory to
    /// allow for async mem copies.
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
