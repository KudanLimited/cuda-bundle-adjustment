#pragma once

#include "macro.h"

namespace cugo
{

/** @brief Camera parameters struct.
 */
struct CUGO_API Camera
{
    /// focal length x (pixel)
    double fx;
    /// focal length y (pixel)
    double fy;
    /// principal point x (pixel)
    double cx;
    /// principal point y (pixel)
    double cy;
    /// stereo baseline * fx
    double bf;

    Camera() : fx(0.0), fy(0.0), cx(0.0), cy(0.0), bf(0.0) {}

    Camera(double fx, double fy, double cx, double cy, double bf)
        : fx(fx), fy(fy), cx(cx), cy(cy), bf(bf)
    {
    }

    Camera(double fx, double fy, double cx, double cy) : fx(fx), fy(fy), cx(cx), cy(cy), bf(1.0)
    {}

    Camera(float fx, float fy, float cx, float cy, float bf)
        : fx(static_cast<double>(fx)),
          fy(static_cast<double>(fy)),
          cx(static_cast<double>(cx)),
          cy(static_cast<double>(cy)),
          bf(static_cast<double>(bf))
    {
    }
};

} // namespace cugo