#ifndef VEC2_H
#define VEC2_H

#include "macros.h"
#include "math.h"
#include "constructor.h"

namespace CX_NAMESPACE
{

template<class Type>
class CxVec2T
{
  public:
	/**
	\brief default constructor leaves data uninitialized.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T()
	{
	}

	/**
	\brief zero constructor.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T(CxZERO) : x(Type(0.0)), y(Type(0.0))
	{
	}

	/**
	\brief Assigns scalar parameter to all elements.

	Useful to initialize to zero or one.

	\param[in] a Value to assign to elements.
	*/
	explicit CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T(Type a) : x(a), y(a)
	{
	}

	/**
	\brief Initializes from 2 scalar parameters.

	\param[in] nx Value to initialize X component.
	\param[in] ny Value to initialize Y component.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T(Type nx, Type ny) : x(nx), y(ny)
	{
	}

	/**
	\brief Copy ctor.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T(const CxVec2T& v) : x(v.x), y(v.y)
	{
	}

	// Operators

	/**
	\brief Assignment operator
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T& operator=(const CxVec2T& p)
	{
		x = p.x;
		y = p.y;
		return *this;
	}

	/**
	\brief element access
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type& operator[](unsigned int index)
	{
		CX_ASSERT(index <= 1);
		return reinterpret_cast<Type*>(this)[index];
	}

	/**
	\brief element access
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE const Type& operator[](unsigned int index) const
	{
		CX_ASSERT(index <= 1);
		return reinterpret_cast<const Type*>(this)[index];
	}

	/**
	\brief returns true if the two vectors are exactly equal.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE bool operator==(const CxVec2T& v) const
	{
		return x == v.x && y == v.y;
	}

	/**
	\brief returns true if the two vectors are not exactly equal.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE bool operator!=(const CxVec2T& v) const
	{
		return x != v.x || y != v.y;
	}

	/**
	\brief tests for exact zero vector
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE bool isZero() const
	{
		return x == Type(0.0) && y == Type(0.0);
	}

	/**
	\brief returns true if all 2 elems of the vector are finite (not NAN or INF, etc.)
	*/
	CX_CUDA_CALLABLE CX_INLINE bool isFinite() const
	{
		return CxIsFinite(x) && CxIsFinite(y);
	}

	/**
	\brief is normalized - used by API parameter validation
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE bool isNormalized() const
	{
		const Type unitTolerance = Type(1e-4);
		return isFinite() && CxAbs(magnitude() - Type(1.0)) < unitTolerance;
	}

	/**
	\brief returns the squared magnitude

	Avoids calling CxSqrt()!
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type magnitudeSquared() const
	{
		return x * x + y * y;
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
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T operator-() const
	{
		return CxVec2T(-x, -y);
	}

	/**
	\brief vector addition
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T operator+(const CxVec2T& v) const
	{
		return CxVec2T(x + v.x, y + v.y);
	}

	/**
	\brief vector difference
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T operator-(const CxVec2T& v) const
	{
		return CxVec2T(x - v.x, y - v.y);
	}

	/**
	\brief scalar post-multiplication
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T operator*(Type f) const
	{
		return CxVec2T(x * f, y * f);
	}

	/**
	\brief scalar division
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T operator/(Type f) const
	{
		f = Type(1.0) / f;
		return CxVec2T(x * f, y * f);
	}

	/**
	\brief vector addition
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T& operator+=(const CxVec2T& v)
	{
		x += v.x;
		y += v.y;
		return *this;
	}

	/**
	\brief vector difference
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T& operator-=(const CxVec2T& v)
	{
		x -= v.x;
		y -= v.y;
		return *this;
	}

	/**
	\brief scalar multiplication
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T& operator*=(Type f)
	{
		x *= f;
		y *= f;
		return *this;
	}

	/**
	\brief scalar division
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T& operator/=(Type f)
	{
		f = Type(1.0) / f;
		x *= f;
		y *= f;
		return *this;
	}

	/**
	\brief returns the scalar product of this and other.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type dot(const CxVec2T& v) const
	{
		return x * v.x + y * v.y;
	}

	/** returns a unit vector */
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T getNormalized() const
	{
		const Type m = magnitudeSquared();
		return m > Type(0.0) ? *this * CxRecipSqrt(m) : CxVec2T(Type(0));
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
	\brief a[i] * b[i], for all i.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T multiply(const CxVec2T& a) const
	{
		return CxVec2T(x * a.x, y * a.y);
	}

	/**
	\brief element-wise minimum
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T minimum(const CxVec2T& v) const
	{
		return CxVec2T(CxMin(x, v.x), CxMin(y, v.y));
	}

	/**
	\brief returns MIN(x, y);
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type minElement() const
	{
		return CxMin(x, y);
	}

	/**
	\brief element-wise maximum
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec2T maximum(const CxVec2T& v) const
	{
		return CxVec2T(CxMax(x, v.x), CxMax(y, v.y));
	}

	/**
	\brief returns MAX(x, y);
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type maxElement() const
	{
		return CxMax(x, y);
	}

	Type	x, y;
};

template<class Type>
CX_CUDA_CALLABLE static CX_FORCE_INLINE CxVec2T<Type> operator*(Type f, const CxVec2T<Type>& v)
{
	return CxVec2T<Type>(f * v.x, f * v.y);
}

typedef CxVec2T<float>	CxVec2;
typedef CxVec2T<double>	CxVec2d;

}



#endif //VEC2_H