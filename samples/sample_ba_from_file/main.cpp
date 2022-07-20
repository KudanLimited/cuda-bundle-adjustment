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

#include <ba_types.h>
#include <cuda_graph_optimisation.h>
#include <icp_types.h>
#include <opencv2/core.hpp>
#include <optimisable_graph.h>
#include <robust_kernel.h>

#include <chrono>
#include <iostream>
#include <vector>

static cugo::CudaGraphOptimisation::Ptr readGraph(const std::string& filename);

cugo::PoseVertexSet* poseVertexSet = nullptr;
cugo::LandmarkVertexSet* landmarkVertexSet = nullptr;
cugo::MonoEdgeSet* monoEdgeSet = nullptr;
cugo::StereoEdgeSet* stereoEdgeSet = nullptr;
cugo::PlaneEdgeSet* planeEdgeSet = nullptr;
std::unique_ptr<cugo::RobustKernelCauchy> kernel;

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: sample_ba_from_file input.json" << std::endl;
        return 0;
    }

    // set robust kernel
    constexpr float fivePercent3DofSqrt = 2.7955321f;
    kernel = std::make_unique<cugo::RobustKernelCauchy>();
    kernel->setDelta(fivePercent3DofSqrt);

    poseVertexSet = new cugo::PoseVertexSet(false);
    landmarkVertexSet = new cugo::LandmarkVertexSet(true);

    planeEdgeSet = new cugo::PlaneEdgeSet();

    monoEdgeSet = new cugo::MonoEdgeSet();
    monoEdgeSet->setRobustKernel(kernel.get());

    stereoEdgeSet = new cugo::StereoEdgeSet();
    stereoEdgeSet->setRobustKernel(kernel.get());

    std::cout << "Reading Graph... " << std::flush;

    auto optimizer = readGraph(argv[1]);

    std::cout << "Done." << std::endl << std::endl;

    std::cout << "=== Graph size : " << std::endl;
    std::cout << "num poses      : " << optimizer->nVertices(0) << std::endl;
    std::cout << "num landmarks  : " << optimizer->nVertices(1) << std::endl;

    auto& edgeSets = optimizer->getEdgeSets();
    for (const auto* edgeSet : edgeSets)
    {
        std::cout << "num edges      : " << edgeSet->nedges() << std::endl << std::endl;
    }

    std::cout << "Running BA... " << std::flush;

    const auto t0 = std::chrono::steady_clock::now();

    optimizer->initialize();
    optimizer->optimize(10);

    const auto t1 = std::chrono::steady_clock::now();

    std::cout << "Done." << std::endl << std::endl;

    const auto duration01 =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

    std::cout << "=== Processing time : " << std::endl;
    std::printf("BA total : %.2f[sec]\n\n", duration01);

    for (const auto& [name, value] : optimizer->timeProfile())
        std::printf("%-30s : %8.1f[msec]\n", name.c_str(), 1e3 * value);
    std::cout << std::endl;

    std::cout << "=== Objective function value : " << std::endl;
    auto batch = optimizer->batchStatistics();
    auto stats = batch.get();
    for (const auto& stat : stats)
        std::printf("iter: %2d, chi2: %.1f\n", stat.iteration + 1, stat.chi2);

    return 0;
}

template <typename T, int N>
static inline cugo::maths::Vec<T, N> getArray(const cv::FileNode& node)
{
    cugo::maths::Vec<T, N> arr = {};
    int pos = 0;
    for (const auto& v : node)
    {
        arr[pos] = T(v);
        if (++pos >= N)
            break;
    }
    return arr;
}

static cugo::CudaGraphOptimisation::Ptr readGraph(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    CV_Assert(fs.isOpened());

    auto optimizer = cugo::CudaGraphOptimisation::create();

    // read pose vertices
    for (const auto& node : fs["pose_vertices"])
    {
        const int id = node["id"];
        const int fixed = node["fixed"];
        const auto q = Eigen::Quaterniond(getArray<double, 4>(node["q"]));
        const auto t = getArray<double, 3>(node["t"]);

        cugo::PoseVertex* poseVertex = new cugo::PoseVertex(id, cugo::maths::Se3D(q, t), fixed);
        poseVertexSet->addVertex(poseVertex);
    }
    optimizer->addVertexSet(poseVertexSet);

    // read landmark vertices
    for (const auto& node : fs["landmark_vertices"])
    {
        const int id = node["id"];
        const int fixed = node["fixed"];
        const auto Xw = getArray<double, 3>(node["Xw"]);

        cugo::LandmarkVertex* landmarkVertex = new cugo::LandmarkVertex(id, Xw, fixed);
        landmarkVertexSet->addVertex(landmarkVertex);
    }
    optimizer->addVertexSet(landmarkVertexSet);

    // read monocular edges
    int i = 0;
    for (const auto& node : fs["monocular_edges"])
    {
        const int iP = node["vertexP"];
        const int iL = node["vertexL"];
        const auto measurement = getArray<double, 2>(node["measurement"]);
        const double information = node["information"];

        auto poseVertex = poseVertexSet->getVertex(iP);
        auto landmarkVertex = landmarkVertexSet->getVertex(iL);

        cugo::MonoEdge* monoEdge = new cugo::MonoEdge();
        monoEdge->setVertex(poseVertex, 0);
        monoEdge->setVertex(landmarkVertex, 1);
        monoEdge->setMeasurement(measurement);
        monoEdge->setInformation(information);
        monoEdgeSet->addEdge(monoEdge);
    }

    // read stereo edges
    i = 0;
    for (const auto& node : fs["stereo_edges"])
    {
        const int iP = node["vertexP"];
        const int iL = node["vertexL"];
        const auto measurement = getArray<double, 3>(node["measurement"]);
        const double information = node["information"];

        auto poseVertex = poseVertexSet->getVertex(iP);
        auto landmarkVertex = landmarkVertexSet->getVertex(iL);

        cugo::StereoEdge* stereoEdge = new cugo::StereoEdge();
        stereoEdge->setVertex(poseVertex, 0);
        stereoEdge->setVertex(landmarkVertex, 1);
        stereoEdge->setMeasurement(measurement);
        stereoEdge->setInformation(information);
        stereoEdgeSet->addEdge(stereoEdge);
    }

    // read camera parameters
    cugo::CameraParams camera;
    camera.fx = fs["fx"];
    camera.fy = fs["fy"];
    camera.cx = fs["cx"];
    camera.cy = fs["cy"];
    camera.bf = fs["bf"];

    optimizer->addEdgeSet<cugo::MonoEdgeSet>(monoEdgeSet);
    optimizer->addEdgeSet<cugo::StereoEdgeSet>(stereoEdgeSet);
    optimizer->setCameraPrams(camera);

    // "warm-up" to avoid overhead
    optimizer->initialize();
    optimizer->optimize(1);

    return optimizer;
}
