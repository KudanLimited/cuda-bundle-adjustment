
#ifndef __OPTIMISABLE_GRAPH_H__
#define __OPTIMISABLE_GRAPH_H__

#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "cuda_bundle_adjustment.h"
#include "cuda_block_solver_impl.h"
#include "device_matrix.h"
#include "device_buffer.h"
#include "cuda_block_solver.h"

namespace cuba
{

////////////////////////////////////////////////////////////////////////////////////
// Type alias
////////////////////////////////////////////////////////////////////////////////////

template <class T, int N>
using Array = Eigen::Matrix<T, N, 1>;

template <class T>
using Set = std::unordered_set<T>;

// forward declerations
class BaseEdge;

////////////////////////////////////////////////////////////////////////////////////
// Vertex
////////////////////////////////////////////////////////////////////////////////////
/** @brief Pose vertex struct.
*/
struct PoseVertex
{
	using Quaternion = Eigen::Quaterniond;
	using Rotation = Quaternion;
	using Translation = Array<double, 3>;

	/** @brief The constructor.
	*/
	PoseVertex() : q(Rotation()), t(Translation()), fixed(false), id(-1), iP(-1) {}

	/** @brief The constructor.
	@param id ID of the vertex.
	@param q rotational component of the pose, represented by quaternions.
	@param t translational component of the pose.
	@param fixed if true, the state variables are fixed during optimization.
	*/
	PoseVertex(int id, const Rotation& q, const Translation& t, bool fixed = false)
		: q(q), t(t), fixed(fixed), id(id), iP(-1) {}

	Rotation q;              //!< rotational component of the pose, represented by quaternions.
	Translation t;           //!< translational component of the pose.
	bool fixed;              //!< if true, the state variables are fixed during optimization.
	int id;                  //!< ID of the vertex.
	int iP;                  //!< ID of the vertex (internally used).
	Set<BaseEdge*> edges;    //!< connected edges.
};

/** @brief Landmark vertex struct.
*/
struct LandmarkVertex
{
	using Point3D = Array<double, 3>;

	/** @brief The constructor.
	*/
	LandmarkVertex() : Xw(Point3D()), fixed(false), id(-1), iL(-1) {}

	/** @brief The constructor.
	@param id ID of the vertex.
	@param Xw 3D position of the landmark.
	@param fixed if true, the state variables are fixed during optimization.
	*/
	LandmarkVertex(int id, const Point3D& Xw, bool fixed = false)
		: Xw(Xw), fixed(fixed), id(id), iL(-1) {}

	Point3D Xw;              //!< 3D position of the landmark.
	bool fixed;              //!< if true, the state variables are fixed during optimization.
	int id;                  //!< ID of the vertex.
	int iL;                  //!< ID of the vertex (internally used).
	Set<BaseEdge*> edges;    //!< connected edges.
};

////////////////////////////////////////////////////////////////////////////////////
// Edge
////////////////////////////////////////////////////////////////////////////////////

/** @brief Base edge struct.
*/
struct BaseEdge
{
	using Information = double;

	/** @brief Returns the connected pose vertex.
	*/
	virtual PoseVertex* getPoseVertex() const = 0;

	/** @brief Returns the connected landmark vertex.
	*/
	virtual LandmarkVertex* getLandmarkVertex() const = 0;

	/** @brief Returns the information for this edge
	*/
	virtual Information information() const = 0;

	/** @brief Returns the dimension of measurement.
	*/
	virtual int dim() const = 0;

	virtual void* getMeasurement() = 0;

};

/** @brief Edge with N-dimensional measurement.
@tparam DIM dimension of the measurement vector.
*/
template <int DIM>
struct Edge : BaseEdge
{
	
	using Measurement = Array<double, DIM>;

	/** @brief The constructor.
	*/
	Edge() : measurement(Measurement()), info(Information()),
		poseVertex(nullptr), landmarkVertex(nullptr) {}

	/** @brief the destructor.
	*/
	virtual ~Edge() {}

	/** @brief The constructor.
	@param m measurement vector.
	@param I information matrix.
	@param PoseVertex connected pose vertex.
	@param LandmarkVertex connected landmark vertex.
	*/
	Edge(const Measurement& m, Information I, PoseVertex* poseVertex, LandmarkVertex* landmarkVertex) :
		measurement(m), info(I), poseVertex(poseVertex), landmarkVertex(landmarkVertex) {}

	/** @brief Returns the connected pose vertex.
	*/
	PoseVertex* getPoseVertex() const override { return poseVertex; }

	/** @brief Returns the connected landmark vertex.
	*/
	LandmarkVertex* getLandmarkVertex() const override { return landmarkVertex; }

	Information information() const override { return info; }

	void* getMeasurement() override { return static_cast<void*>(measurement.data()); }

	/** @brief Returns the dimension of measurement.
	*/
	int dim() const override { return DIM; }

	Measurement measurement;
	Information info; //!< information matrix (represented by a scalar for performance).
	PoseVertex* poseVertex;     //!< connected pose vertex.
	LandmarkVertex* landmarkVertex; //!< connected landmark vertex.
};

/** @brief Edge with 2-dimensional measurement (monocular observation).
*/
using MonoEdge = Edge<2>;

/** @brief Edge with 3-dimensional measurement (stereo observation).
*/
using StereoEdge = Edge<3>;


////////////////////////////////////////////////////////////////////////////////////
// EdgeSet
////////////////////////////////////////////////////////////////////////////////////

class BaseEdgeSet
{
public:

	virtual void addEdge(BaseEdge* edge) = 0;

	virtual void removeEdge(BaseEdge* edge) = 0;

	virtual size_t nedges() const = 0;

	virtual std::unordered_set<BaseEdge*> get() const = 0;

	virtual void* getMeasurementData() = 0;

	virtual const int dim() const = 0;

	virtual void initDevice(const int nedges, void* measurements, Scalar* omegas, 
		CudaBlockSolver::PLIndex* edge2PL, uint8_t* edgeFlags, int* edge2Hpl) = 0;

	// device side virtual functions	
	virtual void constructQuadraticForm(const GpuVec4d& qs, GpuPxPBlockVec& Hpp, GpuPx1BlockVec& bp, 
		GpuLxLBlockVec& Hll, GpuLx1BlockVec& bl, GpuHplBlockMat& Hpl) = 0;

	virtual Scalar computeError(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec3d& Xws, Scalar* chi) = 0;
};

/** @brief groups together a set of edges of the same type. 
*/
template <int DIM>
class EdgeSet : public BaseEdgeSet
{
public:	
	
	using Measurement = Array<double, DIM>;

	// cpu side 
	EdgeSet() {}
	virtual ~EdgeSet() {}

	// vitual functions
    void addEdge(BaseEdge* edge) override
	{
		edges.insert(edge);

		edge->getPoseVertex()->edges.insert(edge);
		edge->getLandmarkVertex()->edges.insert(edge);
	}

	void removeEdge(BaseEdge* e) override
	{
		auto poseVertex = e->getPoseVertex();
		if (poseVertex->edges.count(e))
		{
			poseVertex->edges.erase(e);
		}

		auto landmarkVertex = e->getLandmarkVertex();
		if (landmarkVertex->edges.count(e))
		{
			landmarkVertex->edges.erase(e);
		}
		if (edges.count(e))
		{
			edges.erase(e);
		}
	}

	size_t nedges() const override
	{
		return edges.size();
	}

	std::unordered_set<BaseEdge*> get() const override
	{
		return edges;
	}

	void* getMeasurementData() override
	{
		measurements.clear();
		for (auto* edge : edges)
		{
			measurements.emplace_back(*(static_cast<Measurement*>(edge->getMeasurement())));
		}
		return static_cast<void*>(measurements.data());
	}

	const int dim() const override { return DIM; }

protected:

	std::unordered_set<BaseEdge*> edges;
	std::vector<Measurement> measurements;

public:
	// device side
	using ErrorVec = GpuVec<VecNd<DIM>>;
	using MeasurementVec = GpuVec<VecNd<DIM>>;
	
    void constructQuadraticForm(const GpuVec4d& qs,
	GpuPxPBlockVec& Hpp, GpuPx1BlockVec& bp, GpuLxLBlockVec& Hll, GpuLx1BlockVec& bl, GpuHplBlockMat& Hpl) override
    {
        gpu::constructQuadraticForm_<DIM>(d_Xcs, qs, d_errors, d_omegas, d_edge2PL, d_edge2Hpl, d_edgeFlags, Hpp, bp, Hll, bl, Hpl);
    }

    void initDevice(const int nedges, void* measurements, Scalar* omegas, 
		CudaBlockSolver::PLIndex* edge2PL, uint8_t* edgeFlags, int* edge2Hpl) override
    {
        printf("Measurements....");
		for (int i = 0; i < 10; ++i) {
			for (int j = 0; j < DIM; ++j) {
				auto vec = static_cast<Measurement*>(measurements)[i];
				printf("%f, ", vec[j]);
			}
			printf("\n");
		}
		d_measurements.assign(nedges, measurements);
		d_errors.resize(nedges);
		d_omegas.assign(nedges, omegas);
		d_Xcs.resize(nedges);
		d_edge2PL.assign(nedges, edge2PL);
		d_edgeFlags.assign(nedges, edgeFlags);
		d_edge2Hpl.map(nedges, edge2Hpl);
    }

	Scalar computeError(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec3d& Xws, Scalar* chi) override
	{
		return gpu::computeActiveErrors_<DIM>(qs, ts, Xws, d_measurements, d_omegas, d_edge2PL, d_errors, d_Xcs, chi);
	}

protected:
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
	double fx;               //!< focal length x (pixel)
	double fy;               //!< focal length y (pixel)
	double cx;               //!< principal point x (pixel)
	double cy;               //!< principal point y (pixel)
	double bf;               //!< stereo baseline times fx

	/** @brief The constructor.
	*/
	CameraParams() : fx(0), fy(0), cx(0), cy(0), bf(0) {}
};;

////////////////////////////////////////////////////////////////////////////////////
// Type alias
////////////////////////////////////////////////////////////////////////////////////

using Edge2D = MonoEdge;
using Edge3D = StereoEdge;

}

#endif // __OPTIMISABLE_GRAPH_H__
