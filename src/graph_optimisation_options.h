#pragma once

#include "macro.h"

namespace cugo
{

struct CUGO_API GraphOptimisationOptions
{
    /// Specifies whether weights will be added on a per edge conformation or per a
    /// global weight which is stated per edge set. Using the latter gives a performance
    /// benefit due to the reduction in bandwidth as less information is uploaded to the
    /// GPU.
    bool perEdgeInformation = false;

    /// Specifies whether camera parameters will be add on a per edge basis or a global
    /// set of camera parameters applied to all edges.
    bool perEdgeCamera = false;
};

} // namespace cugo