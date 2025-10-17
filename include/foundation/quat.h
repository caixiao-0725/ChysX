#ifndef QUAT_H
#define QUAT_H


#include "vec3.h"
#include "mat33.h"

namespace CX_NAMESPACE
{


template<class Type> class CxMat33T;

/**
\brief This is a quaternion class. For more information on quaternion mathematics
consult a mathematics source on complex numbers.
*/

template<class Type>
class CxQuatT
{
  public:

	/**
	\brief Default constructor, does not do any initialization.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT()
	{
	}

	//! identity constructor
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT(CxIDENTITY) : x(Type(0.0)), y(Type(0.0)), z(Type(0.0)), w(Type(1.0))
	{
	}

	/**
	\brief Constructor from a scalar: sets the real part w to the scalar value, and the imaginary parts (x,y,z) to zero
	*/
	explicit CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT(Type r) : x(Type(0.0)), y(Type(0.0)), z(Type(0.0)), w(r)
	{
	}

	/**
	\brief Constructor. Take note of the order of the elements!
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT(Type nx, Type ny, Type nz, Type nw) : x(nx), y(ny), z(nz), w(nw)
	{
	}

	/**
	\brief Creates from angle-axis representation.

	Axis must be normalized!

	Angle is in radians!

	<b>Unit:</b> Radians
	*/
	CX_CUDA_CALLABLE CX_INLINE CxQuatT(Type angleRadians, const CxVec3T<Type>& unitAxis)
	{
		CX_ASSERT(CxAbs(Type(1.0) - unitAxis.magnitude()) < Type(1e-3));
		const Type a = angleRadians * Type(0.5);

		Type s;
		CxSinCos(a, s, w);
		x = unitAxis.x * s;
		y = unitAxis.y * s;
		z = unitAxis.z * s;
	}

	/**
	\brief Copy ctor.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT(const CxQuatT& v) : x(v.x), y(v.y), z(v.z), w(v.w)
	{
	}

	/**
	\brief Creates from orientation matrix.

	\param[in] m Rotation matrix to extract quaternion from.
	*/
	CX_CUDA_CALLABLE CX_INLINE explicit CxQuatT(const CxMat33T<Type>& m); /* defined in CxMat33.h */

	/**
	\brief returns true if quat is identity
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE bool isIdentity() const
	{
		return x==Type(0.0) && y==Type(0.0) && z==Type(0.0) && w==Type(1.0);
	}

	/**
	\brief returns true if all elements are finite (not NAN or INF, etc.)
	*/
	CX_CUDA_CALLABLE bool isFinite() const
	{
		return CxIsFinite(x) && CxIsFinite(y) && CxIsFinite(z) && CxIsFinite(w);
	}

	/**
	\brief returns true if finite and magnitude is close to unit
	*/
	CX_CUDA_CALLABLE bool isUnit() const
	{
		const Type unitTolerance = Type(1e-3);
		return isFinite() && CxAbs(magnitude() - Type(1.0)) < unitTolerance;
	}

	/**
	\brief returns true if finite and magnitude is reasonably close to unit to allow for some accumulation of error vs
	isValid
	*/
	CX_CUDA_CALLABLE bool isSane() const
	{
		const Type unitTolerance = Type(1e-2);
		return isFinite() && CxAbs(magnitude() - Type(1.0)) < unitTolerance;
	}

	/**
	\brief returns true if the two quaternions are exactly equal
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE bool operator==(const CxQuatT& q) const
	{
		return x == q.x && y == q.y && z == q.z && w == q.w;
	}

	/**
	\brief converts this quaternion to angle-axis representation
	*/
	CX_CUDA_CALLABLE CX_INLINE void toRadiansAndUnitAxis(Type& angle, CxVec3T<Type>& axis) const
	{
		const Type quatEpsilon = Type(1.0e-8);
		const Type s2 = x * x + y * y + z * z;
		if(s2 < quatEpsilon * quatEpsilon) // can't extract a sensible axis
		{
			angle = Type(0.0);
			axis = CxVec3T<Type>(Type(1.0), Type(0.0), Type(0.0));
		}
		else
		{
			const Type s = CxRecipSqrt(s2);
			axis = CxVec3T<Type>(x, y, z) * s;
			angle = CxAbs(w) < quatEpsilon ? Type(CxPi) : CxAtan2(s2 * s, w) * Type(2.0);
		}
	}

	/**
	\brief Gets the angle between this quat and the identity quaternion.

	<b>Unit:</b> Radians
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type getAngle() const
	{
		return CxAcos(w) * Type(2.0);
	}

	/**
	\brief Gets the angle between this quat and the argument

	<b>Unit:</b> Radians
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type getAngle(const CxQuatT& q) const
	{
		return CxAcos(dot(q)) * Type(2.0);
	}

	/**
	\brief This is the squared 4D vector length, should be 1 for unit quaternions.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type magnitudeSquared() const
	{
		return x * x + y * y + z * z + w * w;
	}

	/**
	\brief returns the scalar product of this and other.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type dot(const CxQuatT& v) const
	{
		return x * v.x + y * v.y + z * v.z + w * v.w;
	}

	CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT getNormalized() const
	{
		const Type s = Type(1.0) / magnitude();
		return CxQuatT(x * s, y * s, z * s, w * s);
	}

	CX_CUDA_CALLABLE CX_FORCE_INLINE Type magnitude() const
	{
		return CxSqrt(magnitudeSquared());
	}

	// modifiers:
	/**
	\brief maps to the closest unit quaternion.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type normalize() // convert this CxQuatT to a unit quaternion
	{
		const Type mag = magnitude();
		if(mag != Type(0.0))
		{
			const Type imag = Type(1.0) / mag;

			x *= imag;
			y *= imag;
			z *= imag;
			w *= imag;
		}
		return mag;
	}

	/*
	\brief returns the conjugate.

	\note for unit quaternions, this is the inverse.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT getConjugate() const
	{
		return CxQuatT(-x, -y, -z, w);
	}

	/*
	\brief returns imaginary part.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> getImaginaryPart() const
	{
		return CxVec3T<Type>(x, y, z);
	}

	/** brief computes rotation of x-axis */
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> getBasisVector0() const
	{
		const Type x2 = x * Type(2.0);
		const Type w2 = w * Type(2.0);
		return CxVec3T<Type>((w * w2) - Type(1.0) + x * x2, (z * w2) + y * x2, (-y * w2) + z * x2);
	}

	/** brief computes rotation of y-axis */
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> getBasisVector1() const
	{
		const Type y2 = y * Type(2.0);
		const Type w2 = w * Type(2.0);
		return CxVec3T<Type>((-z * w2) + x * y2, (w * w2) - Type(1.0) + y * y2, (x * w2) + z * y2);
	}

	/** brief computes rotation of z-axis */
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> getBasisVector2() const
	{
		const Type z2 = z * Type(2.0);
		const Type w2 = w * Type(2.0);
		return CxVec3T<Type>((y * w2) + x * z2, (-x * w2) + y * z2, (w * w2) - Type(1.0) + z * z2);
	}

	/**
	rotates passed vec by this (assumed unitary)
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE const CxVec3T<Type> rotate(const CxVec3T<Type>& v) const
	{
		const Type vx = Type(2.0) * v.x;
		const Type vy = Type(2.0) * v.y;
		const Type vz = Type(2.0) * v.z;
		const Type w2 = w * w - Type(0.5);
		const Type dot2 = (x * vx + y * vy + z * vz);
		return CxVec3T<Type>((vx * w2 + (y * vz - z * vy) * w + x * dot2), (vy * w2 + (z * vx - x * vz) * w + y * dot2),
						     (vz * w2 + (x * vy - y * vx) * w + z * dot2));
	}

	/** \brief computes inverse rotation of x-axis */
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> getInvBasisVector0() const
	{
		const Type x2 = x * Type(2.0);
		const Type w2 = w * Type(2.0);
		return CxVec3T<Type>((w * w2) - Type(1.0) + x * x2, (-z * w2) + y * x2, (y * w2) + z * x2);
	}

	/** \brief computes the inverse rotation of the y-axis */
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> getInvBasisVector1() const
	{
		const Type y2 = y * Type(2.0);
		const Type w2 = w * Type(2.0);
		return CxVec3T<Type>((z * w2) + x * y2, (w * w2) - Type(1.0) + y * y2, (-x * w2) + z * y2);
	}

	/** \brief computes the inverse rotation of the z-axis */
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> getInvBasisVector2() const
	{
		const Type z2 = z * Type(2.0);
		const Type w2 = w * Type(2.0);
		return CxVec3T<Type>((-y * w2) + x * z2, (x * w2) + y * z2, (w * w2) - Type(1.0) + z * z2);
	}

	/**
	inverse rotates passed vec by this (assumed unitary)
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE const CxVec3T<Type> rotateInv(const CxVec3T<Type>& v) const
	{
		const Type vx = Type(2.0) * v.x;
		const Type vy = Type(2.0) * v.y;
		const Type vz = Type(2.0) * v.z;
		const Type w2 = w * w - Type(0.5);
		const Type dot2 = (x * vx + y * vy + z * vz);
		return CxVec3T<Type>((vx * w2 - (y * vz - z * vy) * w + x * dot2), (vy * w2 - (z * vx - x * vz) * w + y * dot2),
						    (vz * w2 - (x * vy - y * vx) * w + z * dot2));
	}

	/**
	\brief Assignment operator
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT& operator=(const CxQuatT& p)
	{
		x = p.x;
		y = p.y;
		z = p.z;
		w = p.w;
		return *this;
	}

	CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT& operator*=(const CxQuatT& q)
	{
		const Type tx = w * q.x + q.w * x + y * q.z - q.y * z;
		const Type ty = w * q.y + q.w * y + z * q.x - q.z * x;
		const Type tz = w * q.z + q.w * z + x * q.y - q.x * y;

		w = w * q.w - q.x * x - y * q.y - q.z * z;
		x = tx;
		y = ty;
		z = tz;
		return *this;
	}

	CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT& operator+=(const CxQuatT& q)
	{
		x += q.x;
		y += q.y;
		z += q.z;
		w += q.w;
		return *this;
	}

	CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT& operator-=(const CxQuatT& q)
	{
		x -= q.x;
		y -= q.y;
		z -= q.z;
		w -= q.w;
		return *this;
	}

	CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT& operator*=(const Type s)
	{
		x *= s;
		y *= s;
		z *= s;
		w *= s;
		return *this;
	}

	/** quaternion multiplication */
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT operator*(const CxQuatT& q) const
	{
		return CxQuatT(w * q.x + q.w * x + y * q.z - q.y * z, w * q.y + q.w * y + z * q.x - q.z * x,
		              w * q.z + q.w * z + x * q.y - q.x * y, w * q.w - x * q.x - y * q.y - z * q.z);
	}

	/** quaternion addition */
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT operator+(const CxQuatT& q) const
	{
		return CxQuatT(x + q.x, y + q.y, z + q.z, w + q.w);
	}

	/** quaternion subtraction */
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT operator-() const
	{
		return CxQuatT(-x, -y, -z, -w);
	}

	CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT operator-(const CxQuatT& q) const
	{
		return CxQuatT(x - q.x, y - q.y, z - q.z, w - q.w);
	}

	CX_CUDA_CALLABLE CX_FORCE_INLINE CxQuatT operator*(Type r) const
	{
		return CxQuatT(x * r, y * r, z * r, w * r);
	}

	/** the quaternion elements */
	Type	x, y, z, w;
};

typedef CxQuatT<float>	CxQuat;
typedef CxQuatT<double>	CxQuatd;


} // namespace CX_NAMESPACE


#endif

