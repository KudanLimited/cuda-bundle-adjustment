
#pragma once

#include "arena.h"
#include "async_vector.h"
#include "block_solver.h"
#include "camera.h"
#include "cuda/cuda_block_solver.h"
#include "cuda_graph_optimisation.h"
#include "device_buffer.h"
#include "device_matrix.h"
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


// forward declerations
class BaseEdge;
class BaseRobustKernel;
struct CudaDeviceInfo;

using EdgeContainer = std::unordered_set<BaseEdge*>;

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
    virtual EdgeContainer& getEdges() noexcept = 0;

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
    EdgeContainer& getEdges() noexcept override;
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
    EdgeContainer edges;
};

using PoseVertex = Vertex<Se3D, false>;
using LandmarkVertex = Vertex<Vec3d, true>;

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
template <typename T, typename E>
class VertexSet : public BaseVertexSet
{
public:
    using VertexType = T;
    using EstimateType = E;
    using DeviceType = EstimateType;
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
     * @brief Generate the estimate data.
     */
    void generateEstimateData();

    /**
     * @brief Maps the specified data onto the device allocated space
     * @param d_dataPtr A pointer to the data that will be uploaded
     */
    void mapEstimateData(Scalar* d_dataPtr, const CudaDeviceInfo& deviceInfo);

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
    async_vector<DeviceType>& getEstimates() noexcept;

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
    async_vector<DeviceType> estimates;

    /// used to sync the device estimates with the vertex ids when
    /// updating pose and landmark estimates
    std::vector<size_t> estimateVertexIds;

    /// the vertices associated with this set
    std::vector<VertexType*> vertices;

    /// the number of active vertices
    int activeSize = 0;
};

using PoseVertexSet = VertexSet<PoseVertex, Se3D>;
using LandmarkVertexSet = VertexSet<LandmarkVertex, Vec3d>;


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

    virtual bool anyVerticesNotFixed() const noexcept = 0;

    virtual bool allVerticesNotFixed() const noexcept = 0;

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

    /**
     * @brief Set the camera parameters for this edge.
     * @param camera The camera parameters to set.
     */
    virtual void setCamera(const Camera& camera) noexcept = 0;

    /**
     * @brief Get the camera parameters for this edge.
     * @return The camera parameters used by this edge.
     */
    virtual Camera& getCamera() noexcept = 0;

    virtual void inactivate() noexcept = 0;
    virtual void setActive() noexcept = 0;
    virtual bool isActive() const noexcept = 0;
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

    Edge() : measurement(Measurement()), isActive_(true) {}
    virtual ~Edge() {}

    // virtual functions
    BaseVertex* getVertex(const int index) override;
    void setVertex(BaseVertex* vertex, const int index) override;
    bool allVerticesFixed() const noexcept override;
    bool anyVerticesNotFixed() const noexcept override;
    bool allVerticesNotFixed() const noexcept override;
    int dim() const noexcept override;
    void setInformation(const Information info) noexcept override;
    Information getInformation() noexcept override;
    void setCamera(const Camera& camera) noexcept override;
    Camera& getCamera() noexcept override;
    void inactivate() noexcept override;
    void setActive() noexcept override;
    bool isActive() const noexcept override;

    // non-virtual functions
    template <std::size_t... Ints>
    bool allVerticesFixedNs(std::index_sequence<Ints...>) const
    {
        bool fixed[] = {getVertexN<Ints>()->isFixed()...};
        return std::all_of(std::begin(fixed), std::end(fixed), [](bool value) { return value; });
    }

    template <std::size_t... Ints>
    bool anyVerticesNotFixedNs(std::index_sequence<Ints...>) const
    {
        bool fixed[] = {getVertexN<Ints>()->isFixed()...};
        return std::any_of(std::begin(fixed), std::end(fixed), [](bool value) { return !value; });
    }

    template <std::size_t... Ints>
    bool allVerticesNotFixedNs(std::index_sequence<Ints...>) const
    {
        bool fixed[] = {getVertexN<Ints>()->isFixed()...};
        return std::all_of(std::begin(fixed), std::end(fixed), [](bool value) { return !value; });
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
    Information info_;
    /// Camera paramters for this edge
    Camera camera_;
    /// The vertex types for the edge
    BaseVertex* vertices[VertexSize];
    /// States the activity of this edge - inactive edges are disregarded during graph optimistaion
    bool isActive_;
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
     * @brief The total number of active edges associated with this set.
     * @return The total edge count.
     */
    virtual size_t nedges() const noexcept = 0;

    /**
     * @brief The number of edges associated with this set.
     * @return The active edge count.
     */
    virtual size_t nActiveEdges() const noexcept = 0;


    /**
     * @brief Get a container of edges associated with this set.
     * @return A vector of edges.
     */
    virtual const EdgeContainer& get() noexcept = 0;

    /**
     * @brief Dimension of the measurement vector associated with the edges.
     * This must be identical to the @see Edge DIM
     * @return The dimensions of the measurement vector.
     */
    virtual const int dim() const noexcept = 0;

    /**
     * @brief Initialise the edge vertex set.
     * @param options A @see GraphOptimisationOptions object
     */
    virtual void init(const GraphOptimisationOptions& options) = 0;

    /**
     * @brief Maps the data derived from the @see init call to the device.
     * @param edge2HData A pointer to an edge set hessian data used when Schur complemenet is
     * active.
     * @param stream A CUDA stream object.
     * @param options A @see GraphOptimisationOptions object
     */
    virtual void
    mapDevice(const GraphOptimisationOptions& options, const CudaDeviceInfo& deviceInfo, int* edge2HData = nullptr) = 0;

    virtual void buildHplBlockPos(
        async_vector<HplBlockPos>& hplBlockPos, int edgeOffset) noexcept = 0;
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
    * @brief If the outlier threshold is greater than zero, then any edge outliers determined by the 
    * @p computeErrors kernel, will be removed from the edge container.
    */
    virtual void updateEdges(const CudaDeviceInfo& deviceInfo) noexcept = 0;

    /**
     * @brief Set the Robust Kernel associated with this set (applied to all edges)
     * @param kernel A initialised Robust Kernel class @see BaseRobustKernel
     */
    virtual void setRobustKernel(const RobustKernelType type, Scalar delta) noexcept = 0;

    virtual RobustKernel& getRobustKernel() noexcept = 0;

    /**
     * @brief Sets the information for this edge set.
     * Note: option @p perEdgeInformation must be false when using this function
     */
    virtual void setInformation(const Information info) noexcept = 0;

    /**
     * @brief Set the camera paramters for this set.
     * Note: option @p perEdgeCamera must be false when using this function
     * @param camera A camera struct specifiying the camera paramters to apply.
     */
    virtual void setCamera(const Camera& camera) noexcept = 0;

    /**
     * @brief Get the camera paramters for this set.
     * Note: option @p perEdgeCamera must be false when using this function
     * @return The camera object for this set.
     */
    virtual Camera& getCamera() noexcept = 0;

    /**
     * @brief Returns the global information for this edge set.
     * Note: option @p perEdgeInformation must be false when using this function
     */
    virtual Information getInformation() noexcept = 0;

    /**
    * @brief Return the outlier threshold used to determine edge outliers.
    * @return The outlier threshold - if zero outlier logic not used.
    */
    virtual Scalar getOutlierThreshold() const noexcept = 0;

    virtual uint32_t getOutlierCount() const noexcept = 0;

    virtual bool isDirty() const noexcept = 0;

    virtual void setDirtyState(bool state) noexcept = 0;

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
        const CudaDeviceInfo& deviceInfo)
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
    virtual Scalar computeError(const VertexSetVec& vertexSets, Scalar* chi, const CudaDeviceInfo& deviceInfo)
    {
        return 0;
    }
};

/**
 * @brief Group together a set of edges of the same type.
 * @tparam DIM dimension of the measurement vector.
 * @tparam E The measurement type for this edge 
 * @tparam VertexTypes A varadic template of vertex types associated with this edge.
 */
template <int DIM, typename E, typename... VertexTypes>
class EdgeSet : public BaseEdgeSet
{
public:
    using MeasurementType = E;
    using GpuMeasurementType = MeasurementType;

    static constexpr auto VertexSize = sizeof...(VertexTypes);

    // Note: even though only this edgeset may denote a single vertex type,
    // we always state two integers as this is what is expected on the GPU side
    // If only one vertex is stated, the second element will be ignored.
    using VIndex = std::array<int, 2>;

    // host side
    EdgeSet()
        : activeEdgeSize_(0),
          outlierThreshold(0.0),
          info_(0.0),
          totalBufferSize_(0), 
          isDirty_(true)
    {
    }
    virtual ~EdgeSet() {}

    // vitual functions
    void addEdge(BaseEdge* edge) override;
    void removeEdge(BaseEdge* edge) override;
    size_t nedges() const noexcept override;
    size_t nActiveEdges() const noexcept override;
    void updateEdges(const CudaDeviceInfo& deviceInfo) noexcept override;
    const EdgeContainer& get() noexcept;
    const int dim() const noexcept override;
    void setRobustKernel(const RobustKernelType type, Scalar delta) noexcept override;
    RobustKernel& getRobustKernel() noexcept;
    void clearEdges() noexcept override;
    void setInformation(const Information info) noexcept override;
    Information getInformation() noexcept override;
    void setCamera(const Camera& camera) noexcept override;
    Camera& getCamera() noexcept override;
    Scalar getOutlierThreshold() const noexcept override;
    uint32_t getOutlierCount() const noexcept override;
    bool isDirty() const noexcept override;
    void setDirtyState(bool state) noexcept override;

    // non-virtual functions
    /**
     * @brief Set the threshold in which a error value is considered an outlier
     * @param errorThreshold The outlier threshold value
     */
    void setOutlierThreshold(const Scalar errorThreshold) noexcept;

private:
    /**
     * @brief Converts a camera object to a vector for handling on the device.
     *
     * @param cam Camera object to convert to vector
     * @return A 5d vector containing the camera paramters.
     */
    const Vec5d cameraToVec(const Camera& cam) noexcept;

protected:
    /// The edges associated with this set.
    EdgeContainer edges;
    /// The number of active edges (has a non-fixed vertex)
    size_t activeEdgeSize_;
    /// The robust kernal parameters associated with this edge set.
    RobustKernel kernel;
    /// The threshold in which an error is considered a outlier
    Scalar outlierThreshold;
    /// The toal buffer size for the arena
    size_t totalBufferSize_;
    /// The omega value applied across all edges. This is only used if option @p perEdgeInformation
    /// is false.
    Information info_;
    /// The camera params which which will be applied to all edges in this set. This is only used if
    /// option @p perEdgeCamera is false.
    Camera camera_;
    // outlier members - only used if outlier threshold is greater than zero
    /// Used to download outlier data from device to the host.
    async_vector<int> edgeOutliers_;
    /// The number of outliers counted last frame.
    uint32_t currOutlierCount_;
    /// States whether edges have been declared inactive this run
    bool isDirty_;

public:
    // device side
    using ErrorVec = typename std::conditional<(DIM == 1), GpuVec1d, GpuVec<VecNd<DIM>>>::type;
    using MeasurementVec = GpuVec<GpuMeasurementType>;

    void init(const GraphOptimisationOptions& options) override;

    void buildHplBlockPos(async_vector<HplBlockPos>& hplBlockPos, int edgeOffset) noexcept override;

    void mapDevice(
        const GraphOptimisationOptions& options,
        const CudaDeviceInfo& deviceInfo,
        int* edge2HData = nullptr) override;

    void clearDevice() noexcept override;

protected:
    /// A memory pool for allocating a chunk of memory for all edge set data. Uses pinned memory to
    /// allow for async mem copies.
    Arena arena;
    std::unique_ptr<ArenaPool<Scalar>> omegas;
    std::unique_ptr<ArenaPool<VIndex>> edge2PL;
    std::unique_ptr<ArenaPool<uint8_t>> edgeFlags;
    std::unique_ptr<ArenaPool<MeasurementType>> measurements;
    std::unique_ptr<ArenaPool<Vec5d>> cameras;

    // device
    GpuVec<uint8_t> d_dataBuffer;
    GpuVec3d d_Xcs;
    GpuVec<Scalar> d_omegas;
    GpuVec5d d_cameras;
    MeasurementVec d_measurements;
    ErrorVec d_errors;
    GpuVec2i d_edge2PL;
    GpuVec1b d_edgeFlags;
    GpuVec1i d_edge2Hpl;
    DeviceBuffer<Scalar> d_outlierThreshold;
    GpuVec1i d_outliers;
    GpuVec<Scalar> d_chiValues;
};


#include "optimisable_graph.hpp"

} // namespace cugo
