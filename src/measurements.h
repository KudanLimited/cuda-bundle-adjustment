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
    Vec3 a, b;

    /**
     Distance between a, b.
     */
    ScalarType length;

    /// A matched point, in platform coordinates.
    Vec3 pointP;

    HOST_DEVICE PointToLineMatch() {};

    /**
     Constructor.
     @param start the first point which the line passes through.
     @param finish the second point which the line passes through.
     */
    HOST_DEVICE PointToLineMatch(const Vec3& start, const Vec3& finish) : a(start), b(finish)
    {
        this->length = norm(a - b);
    }

    /**
     @return the starting point with which the line was constructed.
     */
    HOST_DEVICE Vec3 start() const { return a; }

    /**
     @return the end point with which the line was constructed.
     */
    HOST_DEVICE Vec3 end() const { return b; }

    HOST_DEVICE ScalarType normSquared(const Vec3& r)
    {
        return r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    }

    HOST_DEVICE ScalarType norm(const Vec3& r) { return std::sqrt(normSquared(r)); }
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
    HOST_DEVICE PointToPlaneMatch() {};

    /**
     Construct a plane from a unit normal and plane offset.
     @param norm a unit normal (length 1).
     @param offset the plane offset.
     */
    HOST_DEVICE PointToPlaneMatch(const Vec3& norm, Scalar offset, const Vec3& pointP)
    {
        normal = norm;
        originDistance = offset;
        this->pointP = pointP;
    }

    /**
     The plane's normal.
     NOTE: The Plane functions assume this is normalised!
     */
    Vec3 normal;

    /**
     (Signed) distance of the plane from the origin
     This will be positive if the plane normal points away from the origin
     And negative if the plane normal points towards the origin.
     This enables normal * originDistance to represent a point on the plane
     */
    Scalar originDistance;

    Vec3 pointP;
};


} // namespace cugo
