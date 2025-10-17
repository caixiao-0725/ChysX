#ifndef AABB_H
#define AABB_H

#include "vec3.h"
#include "macros.h"
#include "cx_math.h"

namespace CX_NAMESPACE
{

/**
 * \brief Axis-Aligned Bounding Box class
 * 
 * Represents a 3D axis-aligned bounding box defined by minimum and maximum points.
 */
template<typename Type>
class CxAABBT
{
public:
    CxVec3T<Type> _min;
    CxVec3T<Type> _max;

public:
    /**
     * \brief Default constructor - creates an empty/invalid AABB
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxAABBT()
    {
        _max = CxVec3T<Type>(-CX_MAX_F32, -CX_MAX_F32, -CX_MAX_F32);
        _min = CxVec3T<Type>(CX_MAX_F32, CX_MAX_F32, CX_MAX_F32);
    }

    /**
     * \brief Copy constructor
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxAABBT(const CxAABBT& b)
    {
        _min = b._min;
        _max = b._max;
    }

    /**
     * \brief Move constructor
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxAABBT(CxAABBT&& b)
    {
        _min = b._min;
        _max = b._max;
    }

    /**
     * \brief Constructor from min/max coordinates
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxAABBT(const Type& minx, const Type& miny, const Type& minz,
                                              const Type& maxx, const Type& maxy, const Type& maxz)
    {
        _min = CxVec3T<Type>(minx, miny, minz);
        _max = CxVec3T<Type>(maxx, maxy, maxz);
    }

    /**
     * \brief Constructor from a single point
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxAABBT(const CxVec3T<Type>& v)
    {
        _min = _max = v;
    }

    /**
     * \brief Constructor from two points
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxAABBT(const CxVec3T<Type>& v1, const CxVec3T<Type>& v2)
    {
        _min = CxVec3T<Type>(CxMin(v1.x, v2.x), CxMin(v1.y, v2.y), CxMin(v1.z, v2.z));
        _max = CxVec3T<Type>(CxMax(v1.x, v2.x), CxMax(v1.y, v2.y), CxMax(v1.z, v2.z));
    }

    /**
     * \brief Assignment operator
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxAABBT& operator=(const CxAABBT& aabb)
    {
        _min = aabb._min;
        _max = aabb._max;
        return *this;
    }

    /**
	 * \brief Get the maximum point
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type>& GetMax()
    {
        return this->_max;
    }

    /**
     * \brief Get the minimum point
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type>& GetMin()
    {
        return this->_min;
    }

    /**
     * \brief Combine this AABB with a point
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE void Combine(const CxVec3T<Type>& b)
    {
        _min = CxVec3T<Type>(CxMin(_min.x, b.x), CxMin(_min.y, b.y), CxMin(_min.z, b.z));
        _max = CxVec3T<Type>(CxMax(_max.x, b.x), CxMax(_max.y, b.y), CxMax(_max.z, b.z));
    }

    /**
     * \brief Combine this AABB with a point specified by coordinates
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE void Combine(const Type x, const Type y, const Type z)
    {
        _min = CxVec3T<Type>(CxMin(_min.x, x), CxMin(_min.y, y), CxMin(_min.z, z));
        _max = CxVec3T<Type>(CxMax(_max.x, x), CxMax(_max.y, y), CxMax(_max.z, z));
    }

    /**
     * \brief Combine this AABB with another AABB
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE void Combine(const CxAABBT& b)
    {
        _min = CxVec3T<Type>(CxMin(_min.x, b._min.x), CxMin(_min.y, b._min.y), CxMin(_min.z, b._min.z));
        _max = CxVec3T<Type>(CxMax(_max.x, b._max.x), CxMax(_max.y, b._max.y), CxMax(_max.z, b._max.z));
    }

    /**
     * \brief Test if this AABB overlaps with another AABB
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE bool Overlaps(const CxAABBT& b) const
    {
        if (b._min.x > _max.x || b._max.x < _min.x) return false;
        if (b._min.y > _max.y || b._max.y < _min.y) return false;
        if (b._min.z > _max.z || b._max.z < _min.z) return false;
        return true;
    }

    /**
     * \brief Test if this AABB contains a point
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE bool Contains(const CxVec3T<Type>& v) const
    {
        return v.x <= _max.x && v.x >= _min.x &&
               v.y <= _max.y && v.y >= _min.y &&
               v.z <= _max.z && v.z >= _min.z;
    }

    /**
     * \brief Enlarge the AABB by a thickness value on all sides
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE void Enlarge(const Type& thickness)
    {
        _min.x -= thickness; _min.y -= thickness; _min.z -= thickness;
        _max.x += thickness; _max.y += thickness; _max.z += thickness;
    }

    /**
     * \brief Merge two AABBs and calculate quality metric
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE Type Merge(const CxAABBT& a, const CxAABBT& b, Type* qualityMetric)
    {
        _min = CxVec3T<Type>(CxMin(a._min.x, b._min.x), CxMin(a._min.y, b._min.y), CxMin(a._min.z, b._min.z));
        _max = CxVec3T<Type>(CxMax(a._max.x, b._max.x), CxMax(a._max.y, b._max.y), CxMax(a._max.z, b._max.z));
        *qualityMetric = (a.Volume() + b.Volume()) / Volume();
        return *qualityMetric;
    }

    /**
     * \brief Merge two AABBs without quality metric
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE void Merge(const CxAABBT& a, const CxAABBT& b)
    {
        _min = CxVec3T<Type>(CxMin(a._min.x, b._min.x), CxMin(a._min.y, b._min.y), CxMin(a._min.z, b._min.z));
        _max = CxVec3T<Type>(CxMax(a._max.x, b._max.x), CxMax(a._max.y, b._max.y), CxMax(a._max.z, b._max.z));
    }

    /**
     * \brief Get the width (x-axis extent)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE Type Width() const
    {
        return _max.x - _min.x;
    }

    /**
     * \brief Get the height (y-axis extent)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE Type Height() const
    {
        return _max.y - _min.y;
    }

    /**
     * \brief Get the depth (z-axis extent)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE Type Depth() const
    {
        return _max.z - _min.z;
    }

    /**
     * \brief Get the center point of the AABB
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> Center() const
    {
        return CxVec3T<Type>((_min.x + _max.x) * Type(0.5),
                             (_min.y + _max.y) * Type(0.5),
                             (_min.z + _max.z) * Type(0.5));
    }

    /**
     * \brief Get the volume of the AABB
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE Type Volume() const
    {
        return Width() * Height() * Depth();
    }

    /**
     * \brief Get the surface area of the AABB
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE Type SurfaceArea() const
    {
        Type w = Width();
        Type h = Height();
        Type d = Depth();
        return Type(2.0) * (w * h + w * d + h * d);
    }

    /**
     * \brief Reset to empty/invalid state
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE void Empty()
    {
        _max = CxVec3T<Type>(-CX_MAX_F32, -CX_MAX_F32, -CX_MAX_F32);
        _min = CxVec3T<Type>(CX_MAX_F32, CX_MAX_F32, CX_MAX_F32);
    }

    /**
     * \brief Check if the AABB is valid (min <= max)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE bool IsValid() const
    {
        return _min.x <= _max.x && _min.y <= _max.y && _min.z <= _max.z;
    }

    /**
     * \brief Get the diagonal vector of the AABB
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> Diagonal() const
    {
        return _max - _min;
    }

    /**
     * \brief Get the extent (half-size) of the AABB
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> Extent() const
    {
        return (_max - _min) * Type(0.5);
    }

    /**
     * \brief Get the longest axis (0=x, 1=y, 2=z)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE int LongestAxis() const
    {
        CxVec3T<Type> d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }

    /**
     * \brief Get corner point by index (0-7)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> Corner(int index) const
    {
        CX_ASSERT(index >= 0 && index < 8);
        return CxVec3T<Type>(
            (index & 1) ? _max.x : _min.x,
            (index & 2) ? _max.y : _min.y,
            (index & 4) ? _max.z : _min.z
        );
    }
};

// Type definitions
typedef CxAABBT<float> CxAABB;
typedef CxAABBT<double> CxAABBd;

} // namespace CX_NAMESPACE

#endif // AABB_H

