#ifndef MAT33_H
#define MAT33_H

#include "vec3.h"
#include "quat.h"


namespace CX_NAMESPACE
{

/*!
\brief 3x3 matrix class

Some clarifications, as there have been much confusion about matrix formats etc in the past.

Short:
- Matrix have base vectors in columns (vectors are column matrices, 3x1 matrices).
- Matrix is physically stored in column major format
- Matrices are concaternated from left

Long:
Given three base vectors a, b and c the matrix is stored as

|a.x b.x c.x|
|a.y b.y c.y|
|a.z b.z c.z|

Vectors are treated as columns, so the vector v is

|x|
|y|
|z|

And matrices are applied _before_ the vector (pre-multiplication)
v' = M*v

|x'|   |a.x b.x c.x|   |x|   |a.x*x + b.x*y + c.x*z|
|y'| = |a.y b.y c.y| * |y| = |a.y*x + b.y*y + c.y*z|
|z'|   |a.z b.z c.z|   |z|   |a.z*x + b.z*y + c.z*z|


Physical storage and indexing:
To be compatible with popular 3d rendering APIs (read D3d and OpenGL)
the physical indexing is

|0 3 6|
|1 4 7|
|2 5 8|

index = column*3 + row

which in C++ translates to M[column][row]

The mathematical indexing is M_row,column and this is what is used for _-notation
so _12 is 1st row, second column and operator(row, column)!
*/

template<class Type>
class CxMat33T
{
	public:
	//! Default constructor
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat33T()
	{
	}

	//! identity constructor
	CX_CUDA_CALLABLE CX_INLINE CxMat33T(CxIDENTITY) :
		column0(Type(1.0), Type(0.0), Type(0.0)),
		column1(Type(0.0), Type(1.0), Type(0.0)),
		column2(Type(0.0), Type(0.0), Type(1.0))
	{
	}

	//! zero constructor
	CX_CUDA_CALLABLE CX_INLINE CxMat33T(CxZERO) :
		column0(Type(0.0)),
		column1(Type(0.0)),
		column2(Type(0.0))
	{
	}

	//! Construct from three base vectors
	CX_CUDA_CALLABLE CxMat33T(const CxVec3T<Type>& col0, const CxVec3T<Type>& col1, const CxVec3T<Type>& col2) :
		column0(col0),
		column1(col1),
		column2(col2)
	{
	}

	//! constructor from a scalar, which generates a multiple of the identity matrix
	explicit CX_CUDA_CALLABLE CX_INLINE CxMat33T(Type r) :
		column0(r, Type(0.0), Type(0.0)),
		column1(Type(0.0), r, Type(0.0)),
		column2(Type(0.0), Type(0.0), r)
	{
	}

	//! Construct from Type[9]
	explicit CX_CUDA_CALLABLE CX_INLINE CxMat33T(Type values[]) :
		column0(values[0], values[1], values[2]),
		column1(values[3], values[4], values[5]),
		column2(values[6], values[7], values[8])
	{
	}

	//! Construct from a quaternion
	explicit CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat33T(const CxQuatT<Type>& q)
	{
		// PT: TODO: CX-566
		const Type x = q.x;
		const Type y = q.y;
		const Type z = q.z;
		const Type w = q.w;

		const Type x2 = x + x;
		const Type y2 = y + y;
		const Type z2 = z + z;

		const Type xx = x2 * x;
		const Type yy = y2 * y;
		const Type zz = z2 * z;

		const Type xy = x2 * y;
		const Type xz = x2 * z;
		const Type xw = x2 * w;

		const Type yz = y2 * z;
		const Type yw = y2 * w;
		const Type zw = z2 * w;

		column0 = CxVec3T<Type>(Type(1.0) - yy - zz, xy + zw, xz - yw);
		column1 = CxVec3T<Type>(xy - zw, Type(1.0) - xx - zz, yz + xw);
		column2 = CxVec3T<Type>(xz + yw, yz - xw, Type(1.0) - xx - yy);
	}

	//! Copy constructor
	CX_CUDA_CALLABLE CX_INLINE CxMat33T(const CxMat33T& other) :
		column0(other.column0),
		column1(other.column1),
		column2(other.column2)
	{
	}

	//! Assignment operator
	CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat33T& operator=(const CxMat33T& other)
	{
		column0 = other.column0;
		column1 = other.column1;
		column2 = other.column2;
		return *this;
	}

	//! Construct from diagonal, off-diagonals are zero.
	CX_CUDA_CALLABLE CX_INLINE static const CxMat33T createDiagonal(const CxVec3T<Type>& d)
	{
		return CxMat33T(CxVec3T<Type>(d.x, Type(0.0), Type(0.0)),
						CxVec3T<Type>(Type(0.0), d.y, Type(0.0)),
						CxVec3T<Type>(Type(0.0), Type(0.0), d.z));
	}

	//! Computes the outer product of two vectors
	CX_CUDA_CALLABLE CX_INLINE static const CxMat33T outer(const CxVec3T<Type>& a, const CxVec3T<Type>& b)
	{
		return CxMat33T(a * b.x, a * b.y, a * b.z);
	}

	/**
	\brief returns true if the two matrices are exactly equal
	*/
	CX_CUDA_CALLABLE CX_INLINE bool operator==(const CxMat33T& m) const
	{
		return column0 == m.column0 && column1 == m.column1 && column2 == m.column2;
	}

	//! Get transposed matrix
	CX_CUDA_CALLABLE CX_FORCE_INLINE const CxMat33T getTranspose() const
	{
		const CxVec3T<Type> v0(column0.x, column1.x, column2.x);
		const CxVec3T<Type> v1(column0.y, column1.y, column2.y);
		const CxVec3T<Type> v2(column0.z, column1.z, column2.z);

		return CxMat33T(v0, v1, v2);
	}

	//! Get the real inverse
	CX_CUDA_CALLABLE CX_INLINE const CxMat33T getInverse() const
	{
		const Type det = getDeterminant();
		CxMat33T inverse;

		if(det != Type(0.0))
		{
			const Type invDet = Type(1.0) / det;

			inverse.column0.x = invDet * (column1.y * column2.z - column2.y * column1.z);
			inverse.column0.y = invDet * -(column0.y * column2.z - column2.y * column0.z);
			inverse.column0.z = invDet * (column0.y * column1.z - column0.z * column1.y);

			inverse.column1.x = invDet * -(column1.x * column2.z - column1.z * column2.x);
			inverse.column1.y = invDet * (column0.x * column2.z - column0.z * column2.x);
			inverse.column1.z = invDet * -(column0.x * column1.z - column0.z * column1.x);

			inverse.column2.x = invDet * (column1.x * column2.y - column1.y * column2.x);
			inverse.column2.y = invDet * -(column0.x * column2.y - column0.y * column2.x);
			inverse.column2.z = invDet * (column0.x * column1.y - column1.x * column0.y);

			return inverse;
		}
		else
		{
			return CxMat33T(CxIdentity);
		}
	}

	//! Get determinant
	CX_CUDA_CALLABLE CX_INLINE Type getDeterminant() const
	{
		return column0.dot(column1.cross(column2));
	}

	//! Unary minus
	CX_CUDA_CALLABLE CX_INLINE const CxMat33T operator-() const
	{
		return CxMat33T(-column0, -column1, -column2);
	}

	//! Add
	CX_CUDA_CALLABLE CX_INLINE const CxMat33T operator+(const CxMat33T& other) const
	{
		return CxMat33T(column0 + other.column0, column1 + other.column1, column2 + other.column2);
	}

	//! Subtract
	CX_CUDA_CALLABLE CX_INLINE const CxMat33T operator-(const CxMat33T& other) const
	{
		return CxMat33T(column0 - other.column0, column1 - other.column1, column2 - other.column2);
	}

	//! Scalar multiplication
	CX_CUDA_CALLABLE CX_INLINE const CxMat33T operator*(Type scalar) const
	{
		return CxMat33T(column0 * scalar, column1 * scalar, column2 * scalar);
	}

	template<class Type2>
	CX_CUDA_CALLABLE CX_INLINE friend CxMat33T<Type2> operator*(Type2, const CxMat33T<Type2>&);

	//! Matrix vector multiplication (returns 'this->transform(vec)')
	CX_CUDA_CALLABLE CX_INLINE const CxVec3T<Type> operator*(const CxVec3T<Type>& vec) const
	{
		return transform(vec);
	}

	// a <op>= b operators

	//! Matrix multiplication
	CX_CUDA_CALLABLE CX_FORCE_INLINE const CxMat33T operator*(const CxMat33T& other) const
	{
		// Rows from this <dot> columns from other
		// column0 = transform(other.column0) etc
		return CxMat33T(transform(other.column0),
						transform(other.column1),
						transform(other.column2));
	}

	//! Equals-add
	CX_CUDA_CALLABLE CX_INLINE CxMat33T& operator+=(const CxMat33T& other)
	{
		column0 += other.column0;
		column1 += other.column1;
		column2 += other.column2;
		return *this;
	}

	//! Equals-sub
	CX_CUDA_CALLABLE CX_INLINE CxMat33T& operator-=(const CxMat33T& other)
	{
		column0 -= other.column0;
		column1 -= other.column1;
		column2 -= other.column2;
		return *this;
	}

	//! Equals scalar multiplication
	CX_CUDA_CALLABLE CX_INLINE CxMat33T& operator*=(Type scalar)
	{
		column0 *= scalar;
		column1 *= scalar;
		column2 *= scalar;
		return *this;
	}

	//! Equals matrix multiplication
	CX_CUDA_CALLABLE CX_INLINE CxMat33T& operator*=(const CxMat33T& other)
	{
		*this = *this * other;
		return *this;
	}

	//! Element access, mathematical way!
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type operator()(CxU32 row, CxU32 col) const
	{
		return (*this)[col][row];
	}

	//! Element access, mathematical way!
	CX_CUDA_CALLABLE CX_FORCE_INLINE Type& operator()(CxU32 row, CxU32 col)
	{
		return (*this)[col][row];
	}

	// Transform etc

	//! Transform vector by matrix, equal to v' = M*v
	CX_CUDA_CALLABLE CX_FORCE_INLINE const CxVec3T<Type> transform(const CxVec3T<Type>& other) const
	{
		return column0 * other.x + column1 * other.y + column2 * other.z;
	}

	//! Transform vector by matrix transpose, v' = M^t*v
	CX_CUDA_CALLABLE CX_INLINE const CxVec3T<Type> transformTranspose(const CxVec3T<Type>& other) const
	{
		return CxVec3T<Type>(column0.dot(other), column1.dot(other), column2.dot(other));
	}

	CX_CUDA_CALLABLE CX_FORCE_INLINE const Type* front() const
	{
		return &column0.x;
	}

	CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type>& operator[](CxU32 num)
	{
		return (&column0)[num];
	}

	CX_CUDA_CALLABLE CX_FORCE_INLINE const CxVec3T<Type>& operator[](CxU32 num) const
	{
		return (&column0)[num];
	}

	// Data, see above for format!

	CxVec3T<Type>	column0, column1, column2; // the three base vectors
};

template<class Type>
CX_CUDA_CALLABLE CX_INLINE CxMat33T<Type> operator*(Type scalar, const CxMat33T<Type>& m)
{
	return CxMat33T<Type>(scalar * m.column0, scalar * m.column1, scalar * m.column2);
}

// implementation from CxQuat.h
template<class Type>
CX_CUDA_CALLABLE CX_INLINE CxQuatT<Type>::CxQuatT(const CxMat33T<Type>& m)
{
	if(m.column2.z < Type(0))
	{
		if(m.column0.x > m.column1.y)
		{
			const Type t = Type(1.0) + m.column0.x - m.column1.y - m.column2.z;
			*this = CxQuatT<Type>(t, m.column0.y + m.column1.x, m.column2.x + m.column0.z, m.column1.z - m.column2.y) * (Type(0.5) / CxSqrt(t));
		}
		else
		{
			const Type t = Type(1.0) - m.column0.x + m.column1.y - m.column2.z;
			*this = CxQuatT<Type>(m.column0.y + m.column1.x, t, m.column1.z + m.column2.y, m.column2.x - m.column0.z) * (Type(0.5) / CxSqrt(t));
		}
	}
	else
	{
		if(m.column0.x < -m.column1.y)
		{
			const Type t = Type(1.0) - m.column0.x - m.column1.y + m.column2.z;
			*this = CxQuatT<Type>(m.column2.x + m.column0.z, m.column1.z + m.column2.y, t, m.column0.y - m.column1.x) * (Type(0.5) / CxSqrt(t));
		}
		else
		{
			const Type t = Type(1.0) + m.column0.x + m.column1.y + m.column2.z;
			*this = CxQuatT<Type>(m.column1.z - m.column2.y, m.column2.x - m.column0.z, m.column0.y - m.column1.x, t) * (Type(0.5) / CxSqrt(t));
		}
	}
}

typedef CxMat33T<float>		CxMat33;
typedef CxMat33T<double>	CxMat33d;

	/**
	\brief Sets a rotation matrix around the X axis.
	\param m		[out] output rotation matrix
	\param angle	[in] desired angle
	*/
	CX_INLINE	void CxSetRotX(CxMat33& m, CxReal angle)
	{
		m = CxMat33(CxIdentity);

		CxReal sin, cos;
		CxSinCos(angle, sin, cos);

		m[1][1] = m[2][2] = cos;
		m[1][2] = sin;
		m[2][1] = -sin;
	}

	/**
	\brief Sets a rotation matrix around the Y axis.
	\param m		[out] output rotation matrix
	\param angle	[in] desired angle
	*/
	CX_INLINE	void CxSetRotY(CxMat33& m, CxReal angle)
	{
		m = CxMat33(CxIdentity);

		CxReal sin, cos;
		CxSinCos(angle, sin, cos);

		m[0][0] = m[2][2] = cos;
		m[0][2] = -sin;
		m[2][0] = sin;
	}

	/**
	\brief Sets a rotation matrix around the Z axis.
	\param m		[out] output rotation matrix
	\param angle	[in] desired angle
	*/
	CX_INLINE	void CxSetRotZ(CxMat33& m, CxReal angle)
	{
		m = CxMat33(CxIdentity);

		CxReal sin, cos;
		CxSinCos(angle, sin, cos);

		m[0][0] = m[1][1] = cos;
		m[0][1] = sin;
		m[1][0] = -sin;
	}

	/**
	\brief Returns a rotation quaternion around the X axis.
	\param angle	[in] desired angle
	\return Quaternion that rotates around the desired axis
	*/
	CX_INLINE	CxQuat CxGetRotXQuat(float angle)
	{
		CxMat33 m;
		CxSetRotX(m, angle);
		return CxQuat(m);
	}

	/**
	\brief Returns a rotation quaternion around the Y axis.
	\param angle	[in] desired angle
	\return Quaternion that rotates around the desired axis
	*/
	CX_INLINE	CxQuat CxGetRotYQuat(float angle)
	{
		CxMat33 m;
		CxSetRotY(m, angle);
		return CxQuat(m);
	}

	/**
	\brief Returns a rotation quaternion around the Z axis.
	\param angle	[in] desired angle
	\return Quaternion that rotates around the desired axis
	*/
	CX_INLINE	CxQuat CxGetRotZQuat(float angle)
	{
		CxMat33 m;
		CxSetRotZ(m, angle);
		return CxQuat(m);
	}


} // namespace CX_NAMESPACE


#endif

