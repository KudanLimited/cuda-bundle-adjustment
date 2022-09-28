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
constexpr int ADD_LAMBDA_BLOCK_SIZE = 512;
constexpr int RESTORE_DIAGONAL_BLOCK_SIZE = 1024;
constexpr int COMPUTE_BSCHURE_BLOCK_SIZE = 512;
constexpr int COMPUTE_HSCHURE_BLOCK_SIZE = 256;
constexpr int TWIST_CSR_BLOCK_SIZE = 512;
constexpr int SCHUR_COMP_POST_BLOCK_SIZE = 1024;
constexpr int UPDATE_POSES_BLOCK_SIZE = 512;
constexpr int UPDATE_LANDMARKS_BLOCK_SIZE = 512;

} // namespace gpu
} // namespace cugo
