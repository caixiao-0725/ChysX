// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.

#ifndef VEC4_H
#define VEC4_H

#include "cx_math.h"
#include "vec3.h"

/**
\brief 4 Element vector class.

This is a 4-dimensional vector class with public data members.
*/

namespace CX_NAMESPACE
{


template<class Type>
class CxVec4T
{
  public:
	/**
	\brief default constructor leaves data uninitialized.
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T()
	{
	}

	/**
	\brief zero constructor.
	*/
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec4T(CxZERO) : x(Type(0.0)), y(Type(0.0)), z(Type(0.0)), w(Type(0.0))
	{
	}

	/**
	\brief Assigns scalar parameter to all elements.

	Useful to initialize to zero or one.

	\param[in] a Value to assign to elements.
	*/
	explicit CX_CUDA_CALLABLE CX_INLINE CxVec4T(Type a) : x(a), y(a), z(a), w(a)
	{
	}

	/**
	\brief Initializes from 3 scalar parameters.

	\param[in] nx Value to initialize X component.
	\param[in] ny Value to initialize Y component.
	\param[in] nz Value to initialize Z component.
	\param[in] nw Value to initialize W component.
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T(Type nx, Type ny, Type nz, Type nw) : x(nx), y(ny), z(nz), w(nw)
	{
	}

	/**
	\brief Initializes from 3 scalar parameters.

	\param[in] v Value to initialize the X, Y, and Z components.
	\param[in] nw Value to initialize W component.
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T(const CxVec3T<Type>& v, Type nw) : x(v.x), y(v.y), z(v.z), w(nw)
	{
	}

	/**
	\brief Initializes from an array of scalar parameters.

	\param[in] v Value to initialize with.
	*/
	explicit CX_CUDA_CALLABLE CX_INLINE CxVec4T(const Type v[]) : x(v[0]), y(v[1]), z(v[2]), w(v[3])
	{
	}

	/**
	\brief Copy ctor.
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T(const CxVec4T& v) : x(v.x), y(v.y), z(v.z), w(v.w)
	{
	}

	// Operators

	/**
	\brief Assignment operator
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T& operator=(const CxVec4T& p)
	{
		x = p.x;
		y = p.y;
		z = p.z;
		w = p.w;
		return *this;
	}

	/**
	\brief element access
	*/
	CX_CUDA_CALLABLE CX_INLINE Type& operator[](unsigned int index)
	{
		CX_ASSERT(index <= 3);
		return reinterpret_cast<Type*>(this)[index];
	}

	/**
	\brief element access
	*/
	CX_CUDA_CALLABLE CX_INLINE const Type& operator[](unsigned int index) const
	{
		CX_ASSERT(index <= 3);
		return reinterpret_cast<const Type*>(this)[index];
	}

	/**
	\brief returns true if the two vectors are exactly equal.
	*/
	CX_CUDA_CALLABLE CX_INLINE bool operator==(const CxVec4T& v) const
	{
		return x == v.x && y == v.y && z == v.z && w == v.w;
	}

	/**
	\brief returns true if the two vectors are not exactly equal.
	*/
	CX_CUDA_CALLABLE CX_INLINE bool operator!=(const CxVec4T& v) const
	{
		return x != v.x || y != v.y || z != v.z || w != v.w;
	}

	/**
	\brief tests for exact zero vector
	*/
	CX_CUDA_CALLABLE CX_INLINE bool isZero() const
	{
		return x == Type(0) && y == Type(0) && z == Type(0) && w == Type(0);
	}

	/**
	\brief returns true if all 3 elems of the vector are finite (not NAN or INF, etc.)
	*/
	CX_CUDA_CALLABLE CX_INLINE bool isFinite() const
	{
		return CxIsFinite(x) && CxIsFinite(y) && CxIsFinite(z) && CxIsFinite(w);
	}

	/**
	\brief is normalized - used by API parameter validation
	*/
	CX_CUDA_CALLABLE CX_INLINE bool isNormalized() const
	{
		const Type unitTolerance = Type(1e-4);
		return isFinite() && CxAbs(magnitude() - Type(1.0)) < unitTolerance;
	}

	/**
	\brief returns the squared magnitude

	Avoids calling CxSqrt()!
	*/
	CX_CUDA_CALLABLE CX_INLINE Type magnitudeSquared() const
	{
		return x * x + y * y + z * z + w * w;
	}

	/**
	\brief returns the magnitude
	*/
	CX_CUDA_CALLABLE CX_INLINE Type magnitude() const
	{
		return CxSqrt(magnitudeSquared());
	}

	/**
	\brief negation
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T operator-() const
	{
		return CxVec4T(-x, -y, -z, -w);
	}

	/**
	\brief vector addition
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T operator+(const CxVec4T& v) const
	{
		return CxVec4T(x + v.x, y + v.y, z + v.z, w + v.w);
	}

	/**
	\brief vector difference
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T operator-(const CxVec4T& v) const
	{
		return CxVec4T(x - v.x, y - v.y, z - v.z, w - v.w);
	}

	/**
	\brief scalar post-multiplication
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T operator*(Type f) const
	{
		return CxVec4T(x * f, y * f, z * f, w * f);
	}

	/**
	\brief scalar division
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T operator/(Type f) const
	{
		f = Type(1.0) / f;
		return CxVec4T(x * f, y * f, z * f, w * f);
	}

	/**
	\brief vector addition
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T& operator+=(const CxVec4T& v)
	{
		x += v.x;
		y += v.y;
		z += v.z;
		w += v.w;
		return *this;
	}

	/**
	\brief vector difference
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T& operator-=(const CxVec4T& v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;
		w -= v.w;
		return *this;
	}

	/**
	\brief scalar multiplication
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T& operator*=(Type f)
	{
		x *= f;
		y *= f;
		z *= f;
		w *= f;
		return *this;
	}
	/**
	\brief scalar division
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T& operator/=(Type f)
	{
		f = Type(1.0) / f;
		x *= f;
		y *= f;
		z *= f;
		w *= f;
		return *this;
	}

	/**
	\brief returns the scalar product of this and other.
	*/
	CX_CUDA_CALLABLE CX_INLINE Type dot(const CxVec4T& v) const
	{
		return x * v.x + y * v.y + z * v.z + w * v.w;
	}

	/** returns a unit vector */
	CX_CUDA_CALLABLE CX_INLINE CxVec4T getNormalized() const
	{
		const Type m = magnitudeSquared();
		return m > Type(0.0) ? *this * CxRecipSqrt(m) : CxVec4T(Type(0));
	}

	/**
	\brief normalizes the vector in place
	*/
	CX_CUDA_CALLABLE CX_INLINE Type normalize()
	{
		const Type m = magnitude();
		if(m > Type(0.0))
			*this /= m;
		return m;
	}

	/**
	\brief a[i] * b[i], for all i.
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T multiply(const CxVec4T& a) const
	{
		return CxVec4T(x * a.x, y * a.y, z * a.z, w * a.w);
	}

	/**
	\brief element-wise minimum
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T minimum(const CxVec4T& v) const
	{
		return CxVec4T(CxMin(x, v.x), CxMin(y, v.y), CxMin(z, v.z), CxMin(w, v.w));
	}

	/**
	\brief element-wise maximum
	*/
	CX_CUDA_CALLABLE CX_INLINE CxVec4T maximum(const CxVec4T& v) const
	{
		return CxVec4T(CxMax(x, v.x), CxMax(y, v.y), CxMax(z, v.z), CxMax(w, v.w));
	}

	CX_CUDA_CALLABLE CX_INLINE CxVec3T<Type> getXYZ() const
	{
		return CxVec3T<Type>(x, y, z);
	}

	Type	x, y, z, w;
};

template<class Type>
CX_CUDA_CALLABLE static CX_INLINE CxVec4T<Type> operator*(Type f, const CxVec4T<Type>& v)
{
	return CxVec4T<Type>(f * v.x, f * v.y, f * v.z, f * v.w);
}

typedef CxVec4T<float>	CxVec4;
typedef CxVec4T<double>	CxVec4d;

#if !CX_DOXYGEN
} // namespace physx
#endif

#endif

