#pragma once

#include "device_matrix.h"

namespace cugo
{
namespace gpu
{
////////////////////////////////////////////////////////////////////////////////////
// Type alias
////////////////////////////////////////////////////////////////////////////////////

using PxPBlockPtr = BlockPtr<Scalar, PDIM, PDIM>;
using LxLBlockPtr = BlockPtr<Scalar, LDIM, LDIM>;
using PxLBlockPtr = BlockPtr<Scalar, PDIM, LDIM>;
using Px1BlockPtr = BlockPtr<Scalar, PDIM, 1>;
using Lx1BlockPtr = BlockPtr<Scalar, LDIM, 1>;

////////////////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////////////////
constexpr int BLOCK_ACTIVE_ERRORS = 512;
constexpr int BLOCK_MAX_DIAGONAL = 512;
constexpr int BLOCK_COMPUTE_SCALE = 512;
constexpr int BLOCK_QUADRATIC_FORM = 512;

} // namespace gpu
} // namespace cugo
