#include "optimisable_graph.h"

namespace cuba
{

void PoseVertexSet::mapEstimateData(Scalar* d_dataPtr)
{
    int count = 0;
    std::vector<BaseVertex*> fixedVerticesP;

    for (const auto& [id, poseVertex] : vertexMap)
	{
		if (!poseVertex->isFixed())
		{
			poseVertex->setIndex(count++);
			verticesP.push_back(poseVertex);

            PoseVertex* v = static_cast<PoseVertex*>(poseVertex);
            maths::Se3D estimate = v->getEstimate();
			estimates.emplace_back(Se3D(estimate.r.coeffs().data(), estimate.t.data()));
		}
		else
		{
			fixedVerticesP.push_back(poseVertex);
		}
	}

    activeSize = count;

    for (auto poseVertex : fixedVerticesP)
	{
		poseVertex->setIndex(count++);
		verticesP.push_back(poseVertex);

        PoseVertex* v = static_cast<PoseVertex*>(poseVertex);
        maths::Se3D estimate = v->getEstimate();
		estimates.emplace_back(Se3D(estimate.r.coeffs().data(), estimate.t.data()));
	}

    // upload to the device
    d_estimate.map(estimates.size(), d_dataPtr);
    d_estimate.upload(estimates.data());
}

void PoseVertexSet::finalise()
{
    d_estimate.download(estimates.data());

	for (size_t i = 0; i < verticesP.size(); i++)
	{
		PoseVertex* v = static_cast<PoseVertex*>(verticesP[i]);
        maths::Se3D estimate = v->getEstimate();
        estimates[i].copyTo(estimate.r.coeffs().data(), estimate.t.data());
	}
}

void LandmarkVertexSet::mapEstimateData(Scalar* d_dataPtr)
{
    int count = 0;
    std::vector<BaseVertex*> fixedVerticesL;

    for (const auto& [id, landmarkVertex] : vertexMap)
	{
		if (!landmarkVertex->isFixed())
		{
			landmarkVertex->setIndex(count++);
			verticesL.push_back(landmarkVertex);

            LandmarkVertex* v = static_cast<LandmarkVertex*>(landmarkVertex);
            maths::Vec3d estimate = v->getEstimate();
			estimates.emplace_back(estimate.data());
		}
		else
		{
			fixedVerticesL.push_back(landmarkVertex);
		}
	}

    activeSize = count;

    for (auto landmarkVertex : fixedVerticesL)
	{
		landmarkVertex->setIndex(count++);
		verticesL.push_back(landmarkVertex);

        LandmarkVertex* v = static_cast<LandmarkVertex*>(landmarkVertex);
        maths::Vec3d estimate = v->getEstimate();
		estimates.emplace_back(estimate.data());
	}

    // upload to the device
    d_estimate.map(estimates.size(), d_dataPtr);
    d_estimate.upload(estimates.data());
}

void LandmarkVertexSet::finalise()
{
    d_estimate.download(estimates.data());

    for (size_t i = 0; i < verticesL.size(); i++)
	{
		LandmarkVertex* v = static_cast<LandmarkVertex*>(verticesL[i]);
        maths::Vec3d estimate = v->getEstimate();
        estimates[i].copyTo(estimate.data());
	}
}

BaseVertex* VertexSet::getVertex(const int id) const
{
    auto it = vertexMap.find(id);
    if (it == std::end(vertexMap)) 
    {
        printf("Warning: id: %d not found in vertex map.\n", id);
        return nullptr;
    }
    return vertexMap.at(id);
}

bool VertexSet::removeVertex(BaseVertex* v, BaseEdgeSet* edgeSet, BaseVertexSet* vertexSet)
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

}