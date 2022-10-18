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
    solver_->initialize(edgeSets, vertexSets);
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
        cudaDevice_.startTimingEvent();

        if (iteration == 0)
        {
            solver_->buildStructure(edgeSets, vertexSets);
        }

        const double iniF = solver_->computeErrors(edgeSets, vertexSets);
        F = iniF;

        solver_->buildSystem(edgeSets, vertexSets);

        if (iteration == 0)
        {
            lambda = tau * solver_->maxDiagonal();
        }

        int q = 0;
        double rho = -1.0;
        for (; q < maxq && rho < 0; q++)
        {
            solver_->push();

            solver_->setLambda(lambda);

            const bool success = solver_->solve();

            solver_->update(vertexSets);
        
            const double Fhat = solver_->computeErrors(edgeSets, vertexSets);
            const double scale = solver_->computeScale(lambda) + 1e-3;
            const double Fdiff = Fhat - F;
            rho = success ? (F - Fhat) / scale : -1.0;

            if (rho > 0)
            {
                lambda *= clamp(attenuation(rho), 1.0 / 3.0, 2.0 / 3.0);
                nu = 2.0;
                F = Fhat;
                break;
            }
            else
            {
                lambda *= nu;
                nu *= 2.0;
                solver_->restoreDiagonal();
                solver_->pop();
                if (!std::isfinite(lambda) || Fdiff < 1e-4)
                {
                    break;
                }
            }
        }

        double timeTaken = cudaDevice_.stopTimingEvent();

        stats_.addStat({iteration, F});
        if (verbose)
        {
            printf(
                "iteration= %i;   time(ms): %.4f   chi2= %f;   lambda= %f   rho= "
                "%f	   nedges= %i    levenberg iterations = %i\n",
                iteration,
                timeTaken,
                F,
                lambda,
                rho,
                solver_->nedges(),
                q);
        }

        if (q == maxq || rho < 1e-6 || !std::isfinite(lambda))
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

CudaGraphOptimisationImpl::CudaGraphOptimisationImpl()
    : solver_(std::make_unique<BlockSolver>(options, cudaDevice_))
{
}

CudaGraphOptimisationImpl::CudaGraphOptimisationImpl(GraphOptimisationOptions& options)
    : solver_(std::make_unique<BlockSolver>(options, cudaDevice_)), options(options)
{
}

CudaGraphOptimisationImpl::~CudaGraphOptimisationImpl() {}

CudaGraphOptimisation::Ptr CudaGraphOptimisation::create()
{
    return std::make_unique<CudaGraphOptimisationImpl>();
}

CudaGraphOptimisation::~CudaGraphOptimisation() {}

} // namespace cugo
