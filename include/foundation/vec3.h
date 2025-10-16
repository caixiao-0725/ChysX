#ifndef VEC3_H
#define VEC3_H


#include "cx_math.h"
#include "constructor.h"

namespace CX_NAMESPACE
{

/**
\brief 3 Element vector class.

This is a 3-dimensional vector class with public data members.
*/
template<class Type>
class CxVec3T
{
  public:

	/**
	\brief default constructor leaves data uninitialized.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T()
	{
	}

	/**
	\brief zero constructor.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T(CxZERO) : x(Type(0.0)), y(Type(0.0)), z(Type(0.0))
	{
	}

	/**
	\brief Assigns scalar parameter to all elements.

	Useful to initialize to zero or one.

	\param[in] a Value to assign to elements.
	*/
	explicit CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T(Type a) : x(a), y(a), z(a)
	{
	}

	/**
	\brief Initializes from 3 scalar parameters.

	\param[in] nx Value to initialize X component.
	\param[in] ny Value to initialize Y component.
	\param[in] nz Value to initialize Z component.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T(Type nx, Type ny, Type nz) : x(nx), y(ny), z(nz)
	{
	}

	/**
	\brief Copy ctor.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T(const CxVec3T& v) : x(v.x), y(v.y), z(v.z)
	{
	}

	// Operators

	/**
	\brief Assignment operator
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T& operator=(const CxVec3T& p)
	{
		x = p.x;
		y = p.y;
		z = p.z;
		return *this;
	}

	/**
	\brief element access
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type& operator[](unsigned int index)
	{
		CX_ASSERT(index <= 2);
		return reinterpret_cast<Type*>(this)[index];
	}

	/**
	\brief element access
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE const Type& operator[](unsigned int index) const
	{
		CX_ASSERT(index <= 2);
		return reinterpret_cast<const Type*>(this)[index];
	}

	/**
	\brief returns true if the two vectors are exactly equal.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE bool operator==(const CxVec3T& v) const
	{
		return x == v.x && y == v.y && z == v.z;
	}

	/**
	\brief returns true if the two vectors are not exactly equal.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE bool operator!=(const CxVec3T& v) const
	{
		return x != v.x || y != v.y || z != v.z;
	}

	/**
	\brief tests for exact zero vector
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE bool isZero() const
	{
		return x == Type(0.0) && y == Type(0.0) && z == Type(0.0);
	}

	/**
	\brief returns true if all 3 elems of the vector are finite (not NAN or INF, etc.)
	*/
	CX_CUDA_CALLABLE CX_INLINE bool isFinite() const
	{
		return CxIsFinite(x) && CxIsFinite(y) && CxIsFinite(z);
	}

	/**
	\brief is normalized - used by API parameter validation
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE bool isNormalized() const
	{
		const float unitTolerance = Type(1e-4);	// PT: do we need a different epsilon for float & double?
		return isFinite() && CxAbs(magnitude() - Type(1.0)) < unitTolerance;
	}

	/**
	\brief returns the squared magnitude

	Avoids calling CxSqrt()!
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type magnitudeSquared() const
	{
		return x * x + y * y + z * z;
	}

	/**
	\brief returns the magnitude
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type magnitude() const
	{
		return CxSqrt(magnitudeSquared());
	}

	/**
	\brief negation
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T operator-() const
	{
		return CxVec3T(-x, -y, -z);
	}

	/**
	\brief vector addition
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T operator+(const CxVec3T& v) const
	{
		return CxVec3T(x + v.x, y + v.y, z + v.z);
	}

	/**
	\brief vector difference
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T operator-(const CxVec3T& v) const
	{
		return CxVec3T(x - v.x, y - v.y, z - v.z);
	}

	/**
	\brief scalar post-multiplication
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T operator*(Type f) const
	{
		return CxVec3T(x * f, y * f, z * f);
	}

	/**
	\brief scalar division
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T operator/(Type f) const
	{
		f = Type(1.0) / f;
		return CxVec3T(x * f, y * f, z * f);
	}

	/**
	\brief vector addition
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T& operator+=(const CxVec3T& v)
	{
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	/**
	\brief vector difference
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T& operator-=(const CxVec3T& v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	/**
	\brief scalar multiplication
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T& operator*=(Type f)
	{
		x *= f;
		y *= f;
		z *= f;
		return *this;
	}
	/**
	\brief scalar division
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T& operator/=(Type f)
	{
		f = Type(1.0) / f;
		x *= f;
		y *= f;
		z *= f;
		return *this;
	}

	/**
	\brief returns the scalar product of this and other.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type dot(const CxVec3T& v) const
	{
		return x * v.x + y * v.y + z * v.z;
	}

	/**
	\brief cross product
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T cross(const CxVec3T& v) const
	{
		return CxVec3T(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
	}

	/** returns a unit vector */
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T getNormalized() const
	{
		const Type m = magnitudeSquared();
		return m > Type(0.0) ? *this * CxRecipSqrt(m) : CxVec3T(Type(0));
	}

	/**
	\brief normalizes the vector in place
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type normalize()
	{
		const Type m = magnitude();
		if(m > Type(0.0))
			*this /= m;
		return m;
	}

	/**
	\brief normalizes the vector in place. Does nothing if vector magnitude is under CX_NORMALIZATION_EPSILON.
	Returns vector magnitude if >= CX_NORMALIZATION_EPSILON and 0.0f otherwise.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type normalizeSafe()
	{
		const Type mag = magnitude();
		if(mag < CX_NORMALIZATION_EPSILON)	// PT: do we need a different epsilon for float & double?
			return Type(0.0);
		*this *= Type(1.0) / mag;
		return mag;
	}

	/**
	\brief normalizes the vector in place. Asserts if vector magnitude is under CX_NORMALIZATION_EPSILON.
	returns vector magnitude.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type normalizeFast()
	{
		const Type mag = magnitude();
		CX_ASSERT(mag >= CX_NORMALIZATION_EPSILON);	// PT: do we need a different epsilon for float & double?
		*this *= Type(1.0) / mag;
		return mag;
	}

	/**
	\brief a[i] * b[i], for all i.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T multiply(const CxVec3T& a) const
	{
		return CxVec3T(x * a.x, y * a.y, z * a.z);
	}

	/**
	\brief element-wise minimum
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T minimum(const CxVec3T& v) const
	{
		return CxVec3T(CxMin(x, v.x), CxMin(y, v.y), CxMin(z, v.z));
	}

	/**
	\brief returns MIN(x, y, z);
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type minElement() const
	{
		return CxMin(x, CxMin(y, z));
	}

	/**
	\brief element-wise maximum
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T maximum(const CxVec3T& v) const
	{
		return CxVec3T(CxMax(x, v.x), CxMax(y, v.y), CxMax(z, v.z));
	}

	/**
	\brief returns MAX(x, y, z);
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type maxElement() const
	{
		return CxMax(x, CxMax(y, z));
	}

	/**
	\brief returns absolute values of components;
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T abs() const
	{
		return CxVec3T(CxAbs(x), CxAbs(y), CxAbs(z));
	}

	Type	x, y, z;
};

template<class Type>
CX_CUDA_CALLABLE static CX_FORCE_INLINE CxVec3T<Type> operator*(Type f, const CxVec3T<Type>& v)
{
	return CxVec3T<Type>(f * v.x, f * v.y, f * v.z);
}

typedef CxVec3T<float>	CxVec3;
typedef CxVec3T<double>	CxVec3d;

//! A padded version of CxVec3, to safely load its data using SIMD
class CxVec3Padded : public CxVec3
{
	public:
	CX_FORCE_INLINE	CxVec3Padded()								{}
	CX_FORCE_INLINE	~CxVec3Padded()								{}
	CX_FORCE_INLINE	CxVec3Padded(const CxVec3& p) : CxVec3(p)	{}
	CX_FORCE_INLINE	CxVec3Padded(float f) : CxVec3(f)			{}

	/**
	\brief Assignment operator.
	To fix this:
	error: definition of implicit copy assignment operator for 'CxVec3Padded' is deprecated because it has a user-declared destructor [-Werror,-Wdeprecated]
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3Padded& operator=(const CxVec3Padded& p)
	{
		x = p.x;
		y = p.y;
		z = p.z;
		return *this;
	}

	CxU32	padding;
};
CX_COMPILE_TIME_ASSERT(sizeof(CxVec3Padded) == 16);

typedef CxVec3Padded	CxVec3p;

} // namespace cx


#endif
