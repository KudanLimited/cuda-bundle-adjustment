
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
bool VertexSet<T, E, D>::removeVertex(BaseVertex* v, BaseEdgeSet* edgeSet, BaseVertexSet* vertexSet)
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
