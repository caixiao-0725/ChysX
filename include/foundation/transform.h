#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "quat.h"
#include "vec3.h"
#include "constructor.h"
#include "macros.h"

namespace CX_NAMESPACE
{

// Forward declaration
template<class Type> class CxMat44T;

/**
 * \brief Class representing a rigid euclidean transform as a quaternion and a vector
 * 
 * This represents a rigid transformation (rotation + translation) using a
 * quaternion for rotation and a vector for translation.
 */
template<class Type>
class CxTransformT
{
public:
    CxQuatT<Type> q;  //!< Rotation quaternion
    CxVec3T<Type> p;  //!< Translation vector

public:
    /**
     * \brief Default constructor - leaves data uninitialized
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxTransformT()
    {
    }

    /**
     * \brief Identity constructor
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE explicit CxTransformT(CxIDENTITY) : 
        q(CxIdentity), p(CxZero)
    {
    }

    /**
     * \brief Constructor from position (identity rotation)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE explicit CxTransformT(const CxVec3T<Type>& position) : 
        q(CxIdentity), p(position)
    {
    }

    /**
     * \brief Constructor from orientation (zero translation)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE explicit CxTransformT(const CxQuatT<Type>& orientation) : 
        q(orientation), p(Type(0))
    {
        CX_ASSERT(orientation.isSane());
    }

    /**
     * \brief Constructor from position coordinates and orientation
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxTransformT(Type x, Type y, Type z, 
                                                   CxQuatT<Type> aQ = CxQuatT<Type>(CxIdentity)) : 
        q(aQ), p(x, y, z)
    {
    }

    /**
     * \brief Constructor from position and orientation
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxTransformT(const CxVec3T<Type>& position, 
                                                   const CxQuatT<Type>& orientation) : 
        q(orientation), p(position)
    {
        CX_ASSERT(orientation.isSane());
    }

    /**
     * \brief Constructor from 4x4 matrix
     * Defined in mat44.h
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE explicit CxTransformT(const CxMat44T<Type>& m);

    /**
     * \brief Copy constructor
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxTransformT(const CxTransformT& other)
    {
        p = other.p;
        q = other.q;
    }

    /**
     * \brief Assignment operator
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE void operator=(const CxTransformT& other)
    {
        p = other.p;
        q = other.q;
    }

    /**
     * \brief Returns true if the two transforms are exactly equal
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE bool operator==(const CxTransformT& t) const
    {
        return p == t.p && q == t.q;
    }

    /**
     * \brief Returns true if the two transforms are not equal
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE bool operator!=(const CxTransformT& t) const
    {
        return !(*this == t);
    }

    /**
     * \brief Transform composition
     * Equivalent to matrix multiplication
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxTransformT operator*(const CxTransformT& x) const
    {
        CX_ASSERT(x.isSane());
        return Transform(x);
    }

    /**
     * \brief Equals transform composition
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxTransformT& operator*=(const CxTransformT& other)
    {
        *this = *this * other;
        return *this;
    }

    /**
     * \brief Get inverse transform
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxTransformT GetInverse() const
    {
        CX_ASSERT(IsFinite());
        return CxTransformT(q.rotateInv(-p), q.getConjugate());
    }

    /**
     * \brief Return a normalized transform (quaternion with unit magnitude)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxTransformT GetNormalized() const
    {
        return CxTransformT(p, q.getNormalized());
    }

    /**
     * \brief Transform a point (apply rotation and translation)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> Transform(const CxVec3T<Type>& input) const
    {
        CX_ASSERT(IsFinite());
        return q.rotate(input) + p;
    }

    /**
     * \brief Inverse transform a point
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> TransformInv(const CxVec3T<Type>& input) const
    {
        CX_ASSERT(IsFinite());
        return q.rotateInv(input - p);
    }

    /**
     * \brief Rotate a vector (apply rotation only, no translation)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> Rotate(const CxVec3T<Type>& input) const
    {
        CX_ASSERT(IsFinite());
        return q.rotate(input);
    }

    /**
     * \brief Inverse rotate a vector
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> RotateInv(const CxVec3T<Type>& input) const
    {
        CX_ASSERT(IsFinite());
        return q.rotateInv(input);
    }

    /**
     * \brief Transform another transform (composition: first src, then *this)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxTransformT Transform(const CxTransformT& src) const
    {
        CX_ASSERT(src.isSane());
        CX_ASSERT(isSane());
        // src = [srct, srcr] -> [q*srct + p, q*srcr]
        return CxTransformT(q.rotate(src.p) + p, q * src.q);
    }

    /**
     * \brief Inverse transform another transform (composition: first src, then this->inverse)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxTransformT TransformInv(const CxTransformT& src) const
    {
        CX_ASSERT(src.isSane());
        CX_ASSERT(IsFinite());
        // src = [srct, srcr] -> [q^-1*(srct-p), q^-1*srcr]
        const CxQuatT<Type> qinv = q.getConjugate();
        return CxTransformT(qinv.rotate(src.p - p), qinv * src.q);
    }

    /**
     * \brief Returns true if finite and q is a unit quaternion
     */
    CX_CUDA_CALLABLE bool IsValid() const
    {
        return p.isFinite() && q.isFinite() && q.isUnit();
    }

    /**
     * \brief Returns true if finite and quat magnitude is reasonably close to unit
     * This allows for some accumulation of error vs IsValid()
     */
    CX_CUDA_CALLABLE bool isSane() const
    {
        return IsFinite() && q.isSane();
    }

    /**
     * \brief Returns true if all elements are finite (not NAN or INF, etc.)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE bool IsFinite() const
    {
        return p.isFinite() && q.isFinite();
    }

    /**
     * \brief Get identity transform
     */
    CX_CUDA_CALLABLE static CX_FORCE_INLINE CxTransformT Identity()
    {
        return CxTransformT(CxIdentity);
    }
};

// Type definitions
typedef CxTransformT<float> CxTransform;
typedef CxTransformT<double> CxTransformd;

/**
 * \brief A generic padded & aligned transform class.
 *
 * This can be used for safe faster loads & stores, and faster address computations.
 * Padding bytes can be reused to store useful data if needed.
 */
struct CX_ALIGN(16) CxTransformPadded : CxTransform
{
    CX_FORCE_INLINE CxTransformPadded()
    {
    }

    CX_FORCE_INLINE CxTransformPadded(const CxTransformPadded& other) : CxTransform(other)
    {
    }

    CX_FORCE_INLINE explicit CxTransformPadded(const CxTransform& other) : CxTransform(other)
    {
    }

    CX_FORCE_INLINE explicit CxTransformPadded(CxIDENTITY) : CxTransform(CxIdentity)
    {
    }

    CX_FORCE_INLINE explicit CxTransformPadded(const CxVec3& position) : CxTransform(position)
    {
    }

    CX_FORCE_INLINE explicit CxTransformPadded(const CxQuat& orientation) : CxTransform(orientation)
    {
    }

    CX_FORCE_INLINE CxTransformPadded(const CxVec3& p0, const CxQuat& q0) : CxTransform(p0, q0)
    {
    }

    CX_FORCE_INLINE void operator=(const CxTransformPadded& other)
    {
        p = other.p;
        q = other.q;
    }

    CX_FORCE_INLINE void operator=(const CxTransform& other)
    {
        p = other.p;
        q = other.q;
    }

    CxU32 padding;  //!< Padding for alignment
};
CX_COMPILE_TIME_ASSERT(sizeof(CxTransformPadded) == 32);

typedef CxTransformPadded CxTransform32;

} // namespace CX_NAMESPACE

#endif // TRANSFORM_H
