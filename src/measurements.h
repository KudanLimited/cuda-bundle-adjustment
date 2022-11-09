#pragma once

#include "fixed_vector.h"

#include <cmath>

namespace cugo
{
/**
 A line class representing the geometric representation of an infinite line in some 3D space.
 */
template <typename ScalarType>
class PointToLineMatch
{
public:
    /**
     Start/End of the line, shorthanded to a/b to make the maths easier.
     */
    Vec3d a, b;

    /**
     Distance between a, b.
     */
    ScalarType length;

    /// A matched point, in platform coordinates.
    Vec3d pointP;

    HOST_DEVICE_INLINE PointToLineMatch() {};

    /**
     Constructor.
     @param start the first point which the line passes through.
     @param finish the second point which the line passes through.
     */
    HOST_DEVICE_INLINE PointToLineMatch(const Vec3d& start, const Vec3d& finish) : a(start), b(finish)
    {
        Vec3d dist;
        dist.x = a.x - b.x;
        dist.y = a.y - b.y;
        dist.z = a.z - b.z;
        this->length = norm(dist);
    }

    /**
     @return the starting point with which the line was constructed.
     */
    HOST_DEVICE_INLINE Vec3d start() const { return a; }

    /**
     @return the end point with which the line was constructed.
     */
    HOST_DEVICE_INLINE Vec3d end() const { return b; }

    HOST_DEVICE_INLINE ScalarType normSquared(const Vec3d& r)
    {
        return r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    }

    HOST_DEVICE_INLINE ScalarType norm(const Vec3d& r) { return std::sqrt(normSquared(r)); }
};

template <typename Scalar>
class PointToPlaneMatch
{
public:
    /**
     Unsafe default constructor.
     Used as part of G2O edge initialisation.
     If at all possible, call one of the other constructors.
     */
    HOST_DEVICE_INLINE PointToPlaneMatch() {};

    /**
     Construct a plane from a unit normal and plane offset.
     @param norm a unit normal (length 1).
     @param offset the plane offset.
     */
    HOST_DEVICE_INLINE PointToPlaneMatch(const Vec3d& norm, Scalar offset, const Vec3d& pointP)
    {
        normal = norm;
        originDistance = offset;
        this->pointP = pointP;
    }

    /**
     The plane's normal.
     NOTE: The Plane functions assume this is normalised!
     */
    Vec3d normal;

    /**
     (Signed) distance of the plane from the origin
     This will be positive if the plane normal points away from the origin
     And negative if the plane normal points towards the origin.
     This enables normal * originDistance to represent a point on the plane
     */
    Scalar originDistance;

    Vec3d pointP;
};


} // namespace cugo
