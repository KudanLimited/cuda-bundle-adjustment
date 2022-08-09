
// Vertex functions
template <typename T, bool Marginilised>
void Vertex<T, Marginilised>::addEdge(BaseEdge* edge)
{
    assert(edge != nullptr);
    edges.insert(edge);
}

template <typename T, bool Marginilised>
void Vertex<T, Marginilised>::removeEdge(BaseEdge* edge)
{
    assert(edge != nullptr);
    edges.erase(edge);
}

template <typename T, bool Marginilised>
T& Vertex<T, Marginilised>::getEstimate() noexcept
{
    return estimate;
}

template <typename T, bool Marginilised>
void Vertex<T, Marginilised>::setEstimate(const EstimateType& est) noexcept
{
    estimate = est;
}

template <typename T, bool Marginilised>
Set<BaseEdge*>& Vertex<T, Marginilised>::getEdges() noexcept
{
    return edges;
}

template <typename T, bool Marginilised>
void Vertex<T, Marginilised>::setFixed(bool status) noexcept
{
    fixed = status;
}

template <typename T, bool Marginilised>
bool Vertex<T, Marginilised>::isFixed() const noexcept
{
    return fixed;
}

template <typename T, bool Marginilised>
void Vertex<T, Marginilised>::setId(const int id) noexcept
{
    this->id = id;
}

template <typename T, bool Marginilised>
int Vertex<T, Marginilised>::getId() const noexcept
{
    return id;
}

template <typename T, bool Marginilised>
int Vertex<T, Marginilised>::getIndex() const noexcept
{
    return idx;
}

template <typename T, bool Marginilised>
void Vertex<T, Marginilised>::setIndex(const int idx) noexcept
{
    this->idx = idx;
}

template <typename T, bool Marginilised>
bool Vertex<T, Marginilised>::isMarginilised() const noexcept
{
    return marginilised;
}

template <typename T, bool Marginilised>
void Vertex<T, Marginilised>::clearEdges() noexcept
{
    edges.clear();
}

// VertexSet functions
template <typename T, typename EstimateType, typename DeviceType>
void VertexSet<T, EstimateType, DeviceType>::mapEstimateData(Scalar* d_dataPtr)
{
    int count = 0;
    std::vector<T*> fixedVertices;
    vertices.reserve(vertexMap.size());
    estimates.reserve(vertexMap.size());

    for (const auto& vMap : vertexMap)
    {
        T* vertex = vMap.second;
        assert(vertex != nullptr);
        if (!vertex->isFixed())
        {
            vertex->setIndex(count++);
            vertices.push_back(vertex);

            EstimateType estimate = vertex->getEstimate();
            estimates.emplace_back(DeviceType(estimate));
        }
        else
        {
            fixedVertices.push_back(vertex);
        }
    }

    activeSize = count;

    for (auto vertex : fixedVertices)
    {
        vertex->setIndex(count++);
        vertices.push_back(vertex);

        EstimateType estimate = vertex->getEstimate();
        estimates.emplace_back(DeviceType(estimate));
    }

    // upload to the device
    d_estimate.map(estimates.size(), d_dataPtr);
    d_estimate.upload(estimates.data());
}

template <typename T, typename EstimateType, typename DeviceType>
void VertexSet<T, EstimateType, DeviceType>::finalise()
{
    d_estimate.download(estimates.data());

    for (size_t i = 0; i < vertices.size(); i++)
    {
        estimates[i].copyTo(vertices[i]->getEstimate());
    }
}

template <typename T, typename E, typename D>
T* VertexSet<T, E, D>::getVertex(const int id) const
{
    auto it = vertexMap.find(id);
    if (it == std::end(vertexMap))
    {
        printf("Warning: id: %d not found in vertex map.\n", id);
        return nullptr;
    }
    return vertexMap.at(id);
}

template <typename T, typename E, typename D>
bool VertexSet<T, E, D>::removeVertex(BaseVertex* v, BaseEdgeSet* edgeSet)
{
    auto it = vertexMap.find(v->getId());
    if (it == std::end(vertexMap))
    {
        return false;
    }

    for (auto e : it->second->getEdges())
    {
        edgeSet->removeEdge(e);
    }

    vertexMap.erase(it);
    return true;
}

template <typename T, typename E, typename D>
void VertexSet<T, E, D>::addVertex(T* vertex)
{
    vertexMap.emplace(vertex->getId(), vertex);
}

template <typename T, typename E, typename D>
size_t VertexSet<T, E, D>::size() const noexcept
{
    return vertexMap.size();
}

template <typename T, typename E, typename D>
bool VertexSet<T, E, D>::isMarginilised() const noexcept
{
    return marginilised;
}

template <typename T, typename E, typename D>
size_t VertexSet<T, E, D>::estimateDataSize() const noexcept
{
    return estimates.size();
}

template <typename T, typename E, typename D>
void* VertexSet<T, E, D>::getDeviceEstimateData() noexcept
{
    return static_cast<void*>(d_estimate.data());
}

template <typename T, typename E, typename D>
size_t VertexSet<T, E, D>::getDeviceEstimateSize() noexcept
{
    return d_estimate.size();
}

template <typename T, typename E, typename D>
std::vector<T*>& VertexSet<T, E, D>::get() noexcept
{
    return vertices;
}

template <typename T, typename E, typename D>
GpuVec<D>& VertexSet<T, E, D>::getDeviceEstimates() noexcept
{
    return d_estimate;
}

template <typename T, typename E, typename D>
std::vector<D>& VertexSet<T, E, D>::getEstimates() noexcept
{
    return estimates;
}

template <typename T, typename E, typename D>
int VertexSet<T, E, D>::getActiveSize() const noexcept
{
    return activeSize;
}

template <typename T, typename E, typename D>
void VertexSet<T, E, D>::clearEstimates() noexcept
{
    estimates.clear();
    vertices.clear();
    activeSize = 0;
}

template <typename T, typename E, typename D>
void VertexSet<T, E, D>::clearVertices() noexcept
{
    vertexMap.clear();
}

// edge class functioms
template <int DIM, typename E, typename... VertexTypes>
BaseVertex* Edge<DIM, E, VertexTypes...>::getVertex(const int index)
{
    return vertices[index];
}

template <int DIM, typename E, typename... VertexTypes>
void Edge<DIM, E, VertexTypes...>::setVertex(BaseVertex* vertex, const int index)
{
    vertices[index] = vertex;
}

template <int DIM, typename E, typename... VertexTypes>
bool Edge<DIM, E, VertexTypes...>::allVerticesFixed() const noexcept
{
    return allVerticesFixedNs(std::make_index_sequence<VertexSize>());
}

template <int DIM, typename E, typename... VertexTypes>
int Edge<DIM, E, VertexTypes...>::dim() const noexcept
{
    return DIM;
}

template <int DIM, typename E, typename... VertexTypes>
void Edge<DIM, E, VertexTypes...>::setMeasurement(const Measurement& m) noexcept
{
    measurement = m;
}

template <int DIM, typename E, typename... VertexTypes>
void Edge<DIM, E, VertexTypes...>::setInformation(const Information info) noexcept
{
    info_ = info;
}

template <int DIM, typename E, typename... VertexTypes>
Scalar Edge<DIM, E, VertexTypes...>::getInformation() noexcept
{
    return info_;
}

// EdgeSet functions
template <int DIM, typename E, typename F, typename... VertexTypes>
void EdgeSet<DIM, E, F, VertexTypes...>::addEdge(BaseEdge* edge)
{
    for (int i = 0; i < VertexSize; ++i)
    {
        edge->getVertex(i)->addEdge(edge);
    }
    edges.insert(edge);
}

template <int DIM, typename E, typename F, typename... VertexTypes>
void EdgeSet<DIM, E, F, VertexTypes...>::removeEdge(BaseEdge* edge)
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

template <int DIM, typename E, typename F, typename... VertexTypes>
size_t EdgeSet<DIM, E, F, VertexTypes...>::nedges() const noexcept
{
    return edges.size();
}

template <int DIM, typename E, typename F, typename... VertexTypes>
const std::unordered_set<BaseEdge*>& EdgeSet<DIM, E, F, VertexTypes...>::get() noexcept
{
    return edges;
}

template <int DIM, typename E, typename F, typename... VertexTypes>
void* EdgeSet<DIM, E, F, VertexTypes...>::getHessianBlockPos() noexcept
{
    return hessianBlockPos->data();
}

template <int DIM, typename E, typename F, typename... VertexTypes>
size_t EdgeSet<DIM, E, F, VertexTypes...>::getHessianBlockPosSize() const noexcept
{
    return hessianBlockPos->size();
}

template <int DIM, typename E, typename F, typename... VertexTypes>
const int EdgeSet<DIM, E, F, VertexTypes...>::dim() const noexcept
{
    return DIM;
}

template <int DIM, typename E, typename F, typename... VertexTypes>
void EdgeSet<DIM, E, F, VertexTypes...>::setRobustKernel(BaseRobustKernel* kernel) noexcept
{
    this->kernel = kernel;
}

template <int DIM, typename E, typename F, typename... VertexTypes>
BaseRobustKernel* EdgeSet<DIM, E, F, VertexTypes...>::robustKernel() noexcept
{
    return kernel;
}

template <int DIM, typename E, typename F, typename... VertexTypes>
void EdgeSet<DIM, E, F, VertexTypes...>::setOutlierThreshold(const Scalar errorThreshold) noexcept
{
    this->outlierThreshold = errorThreshold;
}

template <int DIM, typename E, typename F, typename... VertexTypes>
std::vector<int>& EdgeSet<DIM, E, F, VertexTypes...>::outliers()
{
    assert(
        outlierThreshold > 0.0 &&
        "No error threshold set for this edgeSet, thus no outliers will have been calcuated "
        "during graph optimisation.");
    edgeOutliers.resize(edges.size());
    d_outliers.copyTo(edgeOutliers.data());
    return edgeOutliers;
}

template <int DIM, typename E, typename F, typename... VertexTypes>
void EdgeSet<DIM, E, F, VertexTypes...>::clearEdges() noexcept
{
    edges.clear();
}

template <int DIM, typename E, typename F, typename... VertexTypes>
void EdgeSet<DIM, E, F, VertexTypes...>::setInformation(const Information info) noexcept
{
    info_ = info;
}

template <int DIM, typename E, typename F, typename... VertexTypes>
Scalar EdgeSet<DIM, E, F, VertexTypes...>::getInformation() noexcept
{
    return info_;
}

template <int DIM, typename E, typename F, typename... VertexTypes>
void EdgeSet<DIM, E, F, VertexTypes...>::init(
    Arena& hBlockPosArena,
    const int edgeIdOffset,
    cudaStream_t stream,
    bool doSchur,
    const GraphOptimisationOptions& options)
{
    // sanity checks for options
    if (options.perEdgeInformation == true)
    {
        // the edge set weight should not be set if on using the per edge option
        assert(info_ == 0.0);
    }

    size_t edgeSize = edges.size();
    int edgeId = edgeIdOffset;

    totalBufferSize_ =
        sizeof(MeasurementType) * edgeSize + sizeof(VIndex) * edgeSize + sizeof(uint8_t) * edgeSize;

    if (options.perEdgeInformation)
    {
        totalBufferSize_ += sizeof(Scalar) * edgeSize;
    }
    else
    {
        totalBufferSize_ += sizeof(Scalar);
    }

    // allocate more buffer space than needed to reduce the need
    // for resizing.
    arena.resize(totalBufferSize_ * 2);
    measurements = arena.allocate<MeasurementType>(edgeSize);
    edge2PL = arena.allocate<VIndex>(edgeSize);
    edgeFlags = arena.allocate<uint8_t>(edgeSize);

    if (options.perEdgeInformation)
    {
        omega = arena.allocate<Scalar>(edgeSize);
    }

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

        if (options.perEdgeInformation)
        {
            omega->push_back(ScalarCast(edge->getInformation()));
        }

        MeasurementType measurement = *(static_cast<MeasurementType*>(edge->getMeasurement()));
        measurements->push_back(measurement);

        if (VertexSize == 1)
        {
            edgeFlags->push_back(BlockSolver::makeEdgeFlag(edge->getVertex(0)->isFixed(), false));
        }
        else
        {
            edgeFlags->push_back(BlockSolver::makeEdgeFlag(
                edge->getVertex(0)->isFixed(), edge->getVertex(1)->isFixed()));
        }

        ++edgeId;
    }
}

template <int DIM, typename E, typename F, typename... VertexTypes>
void EdgeSet<DIM, E, F, VertexTypes...>::mapDevice(
    int* edge2HData, cudaStream_t stream, const GraphOptimisationOptions& options)
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
    if (options.perEdgeInformation)
    {
        d_omega.offset(d_dataBuffer, edgeSize, omega->bufferOffset());
    }
    else
    {
        d_omega.assignAsync(1, &info_, stream);
    }
}

template <int DIM, typename E, typename F, typename... VertexTypes>
void EdgeSet<DIM, E, F, VertexTypes...>::clearDevice() noexcept
{
    arena.clear();
    edgeOutliers.clear();
}