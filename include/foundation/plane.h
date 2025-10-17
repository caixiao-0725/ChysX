#ifndef PLANE_H
#define PLANE_H

#include "vec3.h"
#include "macros.h"
#include "cx_math.h"

namespace CX_NAMESPACE
{

/**
 * \brief Representation of a plane.
 * 
 * Plane equation used: n.dot(v) + d = 0
 * where n is the normal vector and d is the distance from origin
 */
template<typename Type>
class CxPlaneT
{
public:
    CxVec3T<Type> n;  //!< The normal to the plane
    Type d;           //!< The distance from the origin

public:
    /**
     * \brief Default constructor - leaves data uninitialized
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxPlaneT()
    {
    }

    /**
     * \brief Constructor from normal components and distance
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxPlaneT(Type nx, Type ny, Type nz, Type distance)
        : n(nx, ny, nz), d(distance)
    {
    }

    /**
     * \brief Constructor from a normal vector and a distance
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxPlaneT(const CxVec3T<Type>& normal, Type distance)
        : n(normal), d(distance)
    {
    }

    /**
     * \brief Constructor from a point on the plane and a normal
     * 
     * \param[in] point A point that lies on the plane
     * \param[in] normal The normal vector of the plane
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxPlaneT(const CxVec3T<Type>& point, const CxVec3T<Type>& normal)
        : n(normal), d(-point.dot(n))  // p satisfies normal.dot(p) + d = 0
    {
    }

    /**
     * \brief Constructor from three points
     * 
     * Creates a plane passing through three points.
     * The normal is computed using the right-hand rule.
     * 
     * \param[in] p0 First point
     * \param[in] p1 Second point
     * \param[in] p2 Third point
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxPlaneT(const CxVec3T<Type>& p0, const CxVec3T<Type>& p1, const CxVec3T<Type>& p2)
    {
        n = (p1 - p0).cross(p2 - p0).getNormalized();
        d = -p0.dot(n);
    }

    /**
     * \brief Copy constructor
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxPlaneT(const CxPlaneT& p)
        : n(p.n), d(p.d)
    {
    }

    /**
     * \brief Assignment operator
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxPlaneT& operator=(const CxPlaneT& p)
    {
        n = p.n;
        d = p.d;
        return *this;
    }

    /**
     * \brief Returns true if the two planes are exactly equal
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE bool operator==(const CxPlaneT& p) const
    {
        return n == p.n && d == p.d;
    }

    /**
     * \brief Returns true if the two planes are not exactly equal
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE bool operator!=(const CxPlaneT& p) const
    {
        return n != p.n || d != p.d;
    }

    /**
     * \brief Compute the signed distance from a point to the plane
     * 
     * Positive distance means the point is on the side of the plane
     * pointed to by the normal.
     * 
     * \param[in] p The point to test
     * \return Signed distance from point to plane
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE Type Distance(const CxVec3T<Type>& p) const
    {
        return p.dot(n) + d;
    }

    /**
     * \brief Test if a point lies on the plane (within tolerance)
     * 
     * \param[in] p The point to test
     * \return True if the point is on the plane
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE bool Contains(const CxVec3T<Type>& p) const
    {
        return CxAbs(Distance(p)) < Type(1.0e-7);
    }

    /**
     * \brief Project a point onto the plane
     * 
     * Returns the closest point on the plane to the given point.
     * 
     * \param[in] p The point to project
     * \return The projected point on the plane
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> Project(const CxVec3T<Type>& p) const
    {
        return p - n * Distance(p);
    }

    /**
     * \brief Find an arbitrary point in the plane
     * 
     * \return A point that lies on the plane
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> PointInPlane() const
    {
        return -n * d;
    }

    /**
     * \brief Normalize the plane equation
     * 
     * Ensures the normal vector has unit length.
     * This modifies the plane in-place.
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE void Normalize()
    {
        Type denom = Type(1.0) / n.magnitude();
        n *= denom;
        d *= denom;
    }

    /**
     * \brief Get a normalized copy of this plane
     * 
     * \return A plane with unit normal
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxPlaneT GetNormalized() const
    {
        Type denom = Type(1.0) / n.magnitude();
        return CxPlaneT(n * denom, d * denom);
    }

    /**
     * \brief Flip the plane (negate normal and distance)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE void Flip()
    {
        n = -n;
        d = -d;
    }

    /**
     * \brief Get a flipped copy of this plane
     * 
     * \return A plane with opposite normal and distance
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxPlaneT GetFlipped() const
    {
        return CxPlaneT(-n, -d);
    }

    /**
     * \brief Test if the plane is valid (normal is not zero)
     * 
     * \return True if the plane has a non-zero normal
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE bool IsValid() const
    {
        return !n.isZero();
    }
};

// Type definitions
typedef CxPlaneT<float> CxPlane;
typedef CxPlaneT<double> CxPlaned;

} // namespace CX_NAMESPACE

#endif // PLANE_H
