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

#include "cuda_graph_optimisation.h"

#include "constants.h"
#include "cuda_device.h"
#include "device_buffer.h"
#include "device_matrix.h"
#include "optimisable_graph.h"
#include "profile.h"
#include "sparse_block_matrix.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace cugo
{

EdgeSetVec& CudaGraphOptimisationImpl::getEdgeSets() { return edgeSets; }

size_t CudaGraphOptimisationImpl::nVertices(const int id)
{
    assert(id < vertexSets.size());
    return vertexSets[id]->size();
}

void CudaGraphOptimisationImpl::initialize()
{
    solver_->initialize(edgeSets, vertexSets, streams_);
    stats_.clear();
}

void CudaGraphOptimisationImpl::optimize(int niterations)
{
    const int maxq = 10;
    const double tau = 1e-5;

    double nu = 2.0;
    double lambda = 0.0;
    double F = 0.0;


    // Levenberg-Marquardt iteration
    for (int iteration = 0; iteration < niterations; iteration++)
    {
        auto t0 = get_time_point();

        if (iteration == 0)
        {
            solver_->buildStructure(edgeSets, vertexSets, streams_);
        }

        const double iniF = solver_->computeErrors(edgeSets, vertexSets, streams_);
        F = iniF;

        solver_->buildSystem(edgeSets, vertexSets, streams_);

        if (iteration == 0)
        {
            lambda = tau * solver_->maxDiagonal(streams_);
        }

        int q = 0;
        double rho = -1.0;
        for (; q < maxq && rho < 0; q++)
        {
            solver_->push();

            solver_->setLambda(lambda, streams_);

            const bool success = solver_->solve(streams_);

            solver_->update(vertexSets, streams_);
            solver_->restoreDiagonal(streams_);

            const double Fhat = solver_->computeErrors(edgeSets, vertexSets, streams_);
            const double scale = solver_->computeScale(lambda) + 1e-3;
            rho = success ? (F - Fhat) / scale : -1.0;

            if (rho > 0)
            {
                lambda *= clamp(attenuation(rho), 1.0 / 3, 2.0 / 3);
                nu = 2.0;
                F = Fhat;
                break;
            }
            else
            {
                lambda *= nu;
                nu *= 2.0;
                solver_->pop();
            }
        }

        auto t1 = get_time_point();

        stats_.addStat({iteration, F});
        if (verbose)
        {
            auto duration = get_duration(t0, t1);
            printf(
                "iteration= %i;   time: %.4f   chi2= %f;   lambda= %f   rho= "
                "%f	   nedges= %i\n",
                iteration,
                duration,
                F,
                lambda,
                rho,
                solver_->nedges());
        }

        if (q == maxq || rho <= 0.0 || !std::isfinite(lambda))
        {
            break;
        }
    }

    // remove any outliers from the edgesets
    solver_->updateEdges(edgeSets);

    solver_->finalize(vertexSets);
}

void CudaGraphOptimisationImpl::clearEdgeSets() { edgeSets.clear(); }

void CudaGraphOptimisationImpl::clearVertexSets() { vertexSets.clear(); }

BatchStatistics& CudaGraphOptimisationImpl::batchStatistics() { return stats_; }

const TimeProfile& CudaGraphOptimisationImpl::timeProfile() { return timeProfile_; }

void CudaGraphOptimisationImpl::initCuda()
{
    deviceId_ = findCudaDevice();
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp_, deviceId_));

#ifndef USE_ZERO_COPY
    for (int i = 0; i < streams_.size(); ++i)
    {
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }
#else
    // Set flag to enable zero copy access
    cudaSetDeviceFlags(cudaDeviceMapHost);

    // use default stream if using zero copy as copying data to the device
    // async is no longer required.
    for (int i = 0; i < streams_.size(); ++i)
    {
        streams_[i] = 0;
    }
#endif
}

CudaGraphOptimisationImpl::CudaGraphOptimisationImpl()
    : solver_(std::make_unique<BlockSolver>(options))
{
    initCuda();
}

CudaGraphOptimisationImpl::CudaGraphOptimisationImpl(GraphOptimisationOptions& options)
    : solver_(std::make_unique<BlockSolver>(options)), options(options)
{
    initCuda();
}

CudaGraphOptimisationImpl::~CudaGraphOptimisationImpl() {}

CudaGraphOptimisation::Ptr CudaGraphOptimisation::create()
{
    return std::make_unique<CudaGraphOptimisationImpl>();
}

CudaGraphOptimisation::~CudaGraphOptimisation() {}

} // namespace cugo
