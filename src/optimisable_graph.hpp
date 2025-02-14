
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
EdgeContainer& Vertex<T, Marginilised>::getEdges() noexcept
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
template <typename T, typename EstimateType>
void VertexSet<T, EstimateType>::generateEstimateData()
{
    int count = 0;
    std::vector<T*> fixedVertices;
    std::vector<size_t> fixedIds;

    vertices.reserve(vertexMap.size());
    estimates.reserve(vertexMap.size());
    estimateVertexIds.reserve(vertexMap.size());

    for (const auto& [id, vertex] : vertexMap)
    {
        assert(vertex != nullptr);
        if (CUGO_LIKELY(!vertex->isFixed()))
        {
            vertex->setIndex(count++);
            vertices.push_back(vertex);

            EstimateType estimate = vertex->getEstimate();
            estimates.push_back(estimate);
            estimateVertexIds.emplace_back(id);
        }
        else
        {
            fixedVertices.push_back(vertex);
            fixedIds.emplace_back(id);
        }
    }

    activeSize = count;

    for (int i = 0; i < fixedVertices.size(); ++i)
    {
        T* vertex = fixedVertices[i];
        vertex->setIndex(count++);
        vertices.push_back(vertex);

        EstimateType estimate = vertex->getEstimate();
        estimates.push_back(DeviceType(estimate));
        estimateVertexIds.emplace_back(fixedIds[i]);
    }
}

template <typename T, typename EstimateType>
void VertexSet<T, EstimateType>::mapEstimateData(
    Scalar* d_dataPtr, const CudaDeviceInfo& deviceInfo)
{
    // upload to the device
    d_estimate.map(estimates.size(), d_dataPtr);
    d_estimate.uploadAsync(estimates.data(), deviceInfo.stream);
}

template <typename T, typename EstimateType>
void VertexSet<T, EstimateType>::finalise()
{
    d_estimate.download(estimates.data());

    assert(estimates.size() == estimateVertexIds.size());
    for (size_t i = 0; i < estimateVertexIds.size(); i++)
    {
#ifndef NDEBUG
        if (vertexMap.find(estimateVertexIds[i]) == std::end(vertexMap))
        {
            assert(1);
        }
#endif
        T* vertex = vertexMap[estimateVertexIds[i]];
        vertex->setEstimate(estimates[i]);
    }
}

template <typename T, typename E>
T* VertexSet<T, E>::getVertex(const int id) const
{
#ifndef NDEBUG
    auto it = vertexMap.find(id);
    if (it == std::end(vertexMap))
    {
        printf("Warning: id: %d not found in vertex map.\n", id);
        return nullptr;
    }
#endif
    return vertexMap.at(id);
}

template <typename T, typename E>
bool VertexSet<T, E>::removeVertex(BaseVertex* v, BaseEdgeSet* edgeSet)
{
    auto it = vertexMap.find(v->getId());
    if (CUGO_UNLIKELY(it == std::end(vertexMap)))
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

template <typename T, typename E>
void VertexSet<T, E>::addVertex(T* vertex)
{
    vertexMap.emplace(vertex->getId(), vertex);
}

template <typename T, typename E>
size_t VertexSet<T, E>::size() const noexcept
{
    return vertexMap.size();
}

template <typename T, typename E>
bool VertexSet<T, E>::isMarginilised() const noexcept
{
    return marginilised;
}

template <typename T, typename E>
size_t VertexSet<T, E>::estimateDataSize() const noexcept
{
    return estimates.size();
}

template <typename T, typename E>
void* VertexSet<T, E>::getDeviceEstimateData() noexcept
{
    return static_cast<void*>(d_estimate.data());
}

template <typename T, typename E>
size_t VertexSet<T, E>::getDeviceEstimateSize() noexcept
{
    return d_estimate.size();
}

template <typename T, typename E>
std::vector<T*>& VertexSet<T, E>::get() noexcept
{
    return vertices;
}

template <typename T, typename E>
GpuVec<E>& VertexSet<T, E>::getDeviceEstimates() noexcept
{
    return d_estimate;
}

template <typename T, typename E>
async_vector<E>& VertexSet<T, E>::getEstimates() noexcept
{
    return estimates;
}

template <typename T, typename E>
int VertexSet<T, E>::getActiveSize() const noexcept
{
    return activeSize;
}

template <typename T, typename E>
void VertexSet<T, E>::clearEstimates() noexcept
{
    estimates.clear();
    estimateVertexIds.clear();
    vertices.clear();
    activeSize = 0;
}

template <typename T, typename E>
void VertexSet<T, E>::clearVertices() noexcept
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
bool Edge<DIM, E, VertexTypes...>::anyVerticesNotFixed() const noexcept
{
    return anyVerticesNotFixedNs(std::make_index_sequence<VertexSize>());
}

template <int DIM, typename E, typename... VertexTypes>
bool Edge<DIM, E, VertexTypes...>::allVerticesNotFixed() const noexcept
{
    return allVerticesNotFixedNs(std::make_index_sequence<VertexSize>());
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

template <int DIM, typename E, typename... VertexTypes>
void Edge<DIM, E, VertexTypes...>::setCamera(const Camera& camera) noexcept
{
    camera_ = camera;
}

template <int DIM, typename E, typename... VertexTypes>
Camera& Edge<DIM, E, VertexTypes...>::getCamera() noexcept
{
    return camera_;
}

template <int DIM, typename E, typename... VertexTypes>
void Edge<DIM, E, VertexTypes...>::inactivate() noexcept
{
    isActive_ = false;
}

template <int DIM, typename E, typename... VertexTypes>
void Edge<DIM, E, VertexTypes...>::setActive() noexcept
{
    isActive_ = true;
}

template <int DIM, typename E, typename... VertexTypes>
bool Edge<DIM, E, VertexTypes...>::isActive() const noexcept
{
    return isActive_;
}

// EdgeSet functions
template <int DIM, typename E, typename... VertexTypes>
void EdgeSet<DIM, E, VertexTypes...>::addEdge(BaseEdge* edge)
{
    for (int i = 0; i < VertexSize; ++i)
    {
        edge->getVertex(i)->addEdge(edge);
    }
    edges.insert(edge);
}

template <int DIM, typename E, typename... VertexTypes>
void EdgeSet<DIM, E, VertexTypes...>::removeEdge(BaseEdge* edge)
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

template <int DIM, typename E, typename... VertexTypes>
size_t EdgeSet<DIM, E, VertexTypes...>::nedges() const noexcept
{
    return edges.size();
}

template <int DIM, typename E, typename... VertexTypes>
size_t EdgeSet<DIM, E, VertexTypes...>::nActiveEdges() const noexcept
{
    return activeEdgeSize_;
}

template <int DIM, typename E, typename... VertexTypes>
const EdgeContainer& EdgeSet<DIM, E, VertexTypes...>::get() noexcept
{
    return edges;
}

template <int DIM, typename E, typename... VertexTypes>
const int EdgeSet<DIM, E, VertexTypes...>::dim() const noexcept
{
    return DIM;
}

template <int DIM, typename E, typename... VertexTypes>
void EdgeSet<DIM, E, VertexTypes...>::setRobustKernel(
    const RobustKernelType type, Scalar delta) noexcept
{
    kernel.create(type, delta);
}

template <int DIM, typename E, typename... VertexTypes>
RobustKernel& EdgeSet<DIM, E, VertexTypes...>::getRobustKernel() noexcept
{
    return kernel;
}

template <int DIM, typename E, typename... VertexTypes>
void EdgeSet<DIM, E, VertexTypes...>::setOutlierThreshold(const Scalar errorThreshold) noexcept
{
    this->outlierThreshold = errorThreshold;
    d_outlierThreshold.assign(1, &outlierThreshold);
}

template <int DIM, typename E, typename... VertexTypes>
Scalar EdgeSet<DIM, E, VertexTypes...>::getOutlierThreshold() const noexcept
{
    return outlierThreshold;
}

template <int DIM, typename E, typename... VertexTypes>
void EdgeSet<DIM, E, VertexTypes...>::clearEdges() noexcept
{
    edges.clear();
    currOutlierCount_ = 0;
    totalBufferSize_ = 0;
    activeEdgeSize_ = 0;
    // Set edges state to dirty to ensure that the Hpl data is built the first time round
    isDirty_ = true;
}

template <int DIM, typename E, typename... VertexTypes>
void EdgeSet<DIM, E, VertexTypes...>::setInformation(const Information info) noexcept
{
    info_ = info;
}

template <int DIM, typename E, typename... VertexTypes>
Scalar EdgeSet<DIM, E, VertexTypes...>::getInformation() noexcept
{
    return info_;
}

template <int DIM, typename E, typename... VertexTypes>
void EdgeSet<DIM, E, VertexTypes...>::setCamera(const Camera& camera) noexcept
{
    camera_ = camera;
}

template <int DIM, typename E, typename... VertexTypes>
Camera& EdgeSet<DIM, E, VertexTypes...>::getCamera() noexcept
{
    return camera_;
}

template <int DIM, typename E, typename... VertexTypes>
const Vec5d EdgeSet<DIM, E, VertexTypes...>::cameraToVec(const Camera& cam) noexcept
{
    Vec5d output;
    output[0] = ScalarCast(cam.fx);
    output[1] = ScalarCast(cam.fy);
    output[2] = ScalarCast(cam.cx);
    output[3] = ScalarCast(cam.cy);
    output[4] = ScalarCast(cam.bf);
    return output;
}

template <int DIM, typename E, typename... VertexTypes>
void EdgeSet<DIM, E, VertexTypes...>::init(const GraphOptimisationOptions& options)
{
    // sanity checks for options
    if (options.perEdgeInformation)
    {
        // the edge set weight should not be set if on using the per edge option
        assert(info_ == 0.0);
    }

    // calculate the number of active edges (either vertex is not fixed)
    // NOTE: The assumption here is that there is either one vertex and this is
    // a pose type or there are two vertices and they are pose and landmark (in that order)
    activeEdgeSize_ = 0;
    for (BaseEdge* edge : edges)
    {
        if (VertexSize == 1)
        {
            if (!edge->getVertex(0)->isFixed())
            {
                ++activeEdgeSize_;
            }
        }
        else
        {
            if (!edge->getVertex(0)->isFixed() || !edge->getVertex(1)->isFixed())
            {
                ++activeEdgeSize_;
            }
        }
    }

    const size_t omegaSize = (options.perEdgeInformation) ? activeEdgeSize_ : 1;
    const size_t cameraSize = (options.perEdgeCamera) ? activeEdgeSize_ : 2;

    totalBufferSize_ = sizeof(MeasurementType) * activeEdgeSize_ +
        sizeof(VIndex) * activeEdgeSize_ + sizeof(uint8_t) * activeEdgeSize_ +
        omegaSize * sizeof(Scalar) + cameraSize * (sizeof(Scalar) * 5);

    // allocate more buffer space than needed to reduce the need
    // for resizing.
    arena.resize(totalBufferSize_ * 2);
    measurements = arena.allocate<MeasurementType>(activeEdgeSize_);
    edge2PL = arena.allocate<VIndex>(activeEdgeSize_);
    edgeFlags = arena.allocate<uint8_t>(activeEdgeSize_);
    omegas = arena.allocate<Scalar>(omegaSize);
    cameras = arena.allocate<Vec5d>(cameraSize);

    if (!options.perEdgeInformation)
    {
        omegas->push_back(ScalarCast(info_));
    }
    if (!options.perEdgeCamera)
    {
        cameras->push_back(cameraToVec(camera_));
    }

    for (BaseEdge* edge : edges)
    {
        bool isActive = false;
        VIndex vec;
        if (VertexSize == 1)
        {
            if (CUGO_LIKELY(!edge->getVertex(0)->isFixed()))
            {
                vec[0] = edge->getVertex(0)->getIndex();
                edgeFlags->push_back(
                    BlockSolver::makeEdgeFlag(edge->getVertex(0)->isFixed(), false));
                isActive = true;
            }
        }
        else
        {
            if (CUGO_LIKELY(!edge->getVertex(0)->isFixed() || !edge->getVertex(1)->isFixed()))
            {
                vec[0] = edge->getVertex(0)->getIndex();
                vec[1] = edge->getVertex(1)->getIndex();
                edgeFlags->push_back(BlockSolver::makeEdgeFlag(
                    edge->getVertex(0)->isFixed(), edge->getVertex(1)->isFixed()));
                isActive = true;
            }
        }

        if (isActive)
        {
            measurements->push_back(*(static_cast<MeasurementType*>(edge->getMeasurement())));
            edge2PL->push_back(vec);

            if (options.perEdgeInformation)
            {
                omegas->push_back(ScalarCast(edge->getInformation()));
            }
            if (options.perEdgeCamera)
            {
                cameras->push_back(cameraToVec(edge->getCamera()));
            }
        }
    }
}

template <int DIM, typename E, typename... VertexTypes>
void EdgeSet<DIM, E, VertexTypes...>::mapDevice(
    const GraphOptimisationOptions& options, const CudaDeviceInfo& deviceInfo, int* edge2HData)
{
    // buffers filled by the gpu kernels.
    d_errors.resize(activeEdgeSize_);
    d_Xcs.resize(activeEdgeSize_);
    d_chiValues.resize(activeEdgeSize_);

    d_outliers.resize(activeEdgeSize_);
    d_outliers.fillZero();

    if (edge2HData)
    {
        d_edge2Hpl.map(activeEdgeSize_, edge2HData);
    }

    // The main "mega" buffer which contains all of the data used
    // in optimising the graph - transferring one large buffer async
    // is far more optimal than transferring multiple smaller buffers
    d_dataBuffer.assignAsync(totalBufferSize_, arena.data(), deviceInfo.stream);

    d_edgeFlags.offset(d_dataBuffer, edgeFlags->size(), edgeFlags->bufferOffset());
    d_edge2PL.offset(d_dataBuffer, edge2PL->size(), edge2PL->bufferOffset());
    d_measurements.offset(d_dataBuffer, measurements->size(), measurements->bufferOffset());
    d_omegas.offset(d_dataBuffer, omegas->size(), omegas->bufferOffset());
    d_cameras.offset(d_dataBuffer, cameras->size(), cameras->bufferOffset());
}

template <int DIM, typename E, typename... VertexTypes>
void EdgeSet<DIM, E, VertexTypes...>::updateEdges(const CudaDeviceInfo& deviceInfo) noexcept
{
    if (CUGO_UNLIKELY(!edges.size()))
    {
        return;
    }

    if (outlierThreshold > 0.0)
    {
        const size_t nedges = edges.size();
        gpu::computeOutliers(nedges, outlierThreshold, d_chiValues, d_outliers, deviceInfo);

        edgeOutliers_.resize(activeEdgeSize_);
        d_outliers.download(edgeOutliers_.data());

        size_t idx = 0;
        uint32_t outlierCount = 0;

        assert(edgeOutliers_.size() == edges.size());
        for (BaseEdge* edge : edges)
        {
            if (CUGO_UNLIKELY(edgeOutliers_[idx++]))
            {
                edge->inactivate();
                outlierCount++;
            }
        }
        const uint32_t outliersRemovedThisFrame = outlierCount - currOutlierCount_;
        if (outliersRemovedThisFrame > 0)
        {
            // denotes that the number of active edges has changed this frame so
            // update the appropiate buffers
            isDirty_ = true;
        }
        currOutlierCount_ = outlierCount;
    }
}

template <int DIM, typename E, typename... VertexTypes>
void EdgeSet<DIM, E, VertexTypes...>::buildHplBlockPos(
    async_vector<HplBlockPos>& hplBlockPos, int edgeOffset) noexcept
{
    int edgeId = edgeOffset;
    for (BaseEdge* edge : edges)
    {
        if (CUGO_UNLIKELY(edge->allVerticesFixed()))
        {
            continue;
        }

        if (edge->isActive() && edge->allVerticesNotFixed())
        {
            hplBlockPos.push_back(
                {edge->getVertex(0)->getIndex(), edge->getVertex(1)->getIndex(), edgeId});
        }
        ++edgeId;
    }
}

template <int DIM, typename E, typename... VertexTypes>
uint32_t EdgeSet<DIM, E, VertexTypes...>::getOutlierCount() const noexcept
{
    return currOutlierCount_;
}

template <int DIM, typename E, typename... VertexTypes>
uint32_t EdgeSet<DIM, E, VertexTypes...>::getInlierCount() const noexcept
{
    return activeEdgeSize_ - currOutlierCount_;
}

template <int DIM, typename E, typename... VertexTypes>
bool EdgeSet<DIM, E, VertexTypes...>::isDirty() const noexcept
{
    return isDirty_;
}

template <int DIM, typename E, typename... VertexTypes>
void EdgeSet<DIM, E, VertexTypes...>::setDirtyState(bool state) noexcept
{
    isDirty_ = state;
}

template <int DIM, typename E, typename... VertexTypes>
void EdgeSet<DIM, E, VertexTypes...>::clearDevice() noexcept
{
    arena.clear();
}