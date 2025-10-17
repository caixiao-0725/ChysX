#ifndef MAT44_H
#define MAT44_H

#include "quat.h"
#include "vec4.h"
#include "vec3.h"
#include "mat33.h"
#include "constructor.h"
#include "macros.h"

namespace CX_NAMESPACE
{

// Forward declaration
template<class Type> class CxTransformT;

/**
 * \brief 4x4 matrix class
 *
 * This class is layout-compatible with D3D and OpenGL matrices.
 * Matrix is stored in column-major order for GPU compatibility.
 *
 * \see CxMat33T CxTransformT
 */
template<class Type>
class CxMat44T
{
public:
    //! Column vectors
    CxVec4T<Type> column0, column1, column2, column3;

public:
    /**
     * \brief Default constructor - leaves data uninitialized
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T()
    {
    }

    /**
     * \brief Identity constructor
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T(CxIDENTITY) :
        column0(Type(1.0), Type(0.0), Type(0.0), Type(0.0)),
        column1(Type(0.0), Type(1.0), Type(0.0), Type(0.0)),
        column2(Type(0.0), Type(0.0), Type(1.0), Type(0.0)),
        column3(Type(0.0), Type(0.0), Type(0.0), Type(1.0))
    {
    }

    /**
     * \brief Zero constructor
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T(CxZERO) :
        column0(CxZero), column1(CxZero), column2(CxZero), column3(CxZero)
    {
    }

    /**
     * \brief Construct from four 4-vectors
     */
    CX_CUDA_CALLABLE CxMat44T(const CxVec4T<Type>& col0, const CxVec4T<Type>& col1, 
                               const CxVec4T<Type>& col2, const CxVec4T<Type>& col3) :
        column0(col0), column1(col1), column2(col2), column3(col3)
    {
    }

    /**
     * \brief Constructor that generates a multiple of the identity matrix
     */
    explicit CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T(Type r) :
        column0(r, Type(0.0), Type(0.0), Type(0.0)),
        column1(Type(0.0), r, Type(0.0), Type(0.0)),
        column2(Type(0.0), Type(0.0), r, Type(0.0)),
        column3(Type(0.0), Type(0.0), Type(0.0), r)
    {
    }

    /**
     * \brief Construct from three base vectors and a translation
     */
    CX_CUDA_CALLABLE CxMat44T(const CxVec3T<Type>& col0, const CxVec3T<Type>& col1, 
                               const CxVec3T<Type>& col2, const CxVec3T<Type>& col3) :
        column0(col0, Type(0.0)),
        column1(col1, Type(0.0)),
        column2(col2, Type(0.0)),
        column3(col3, Type(1.0))
    {
    }

    /**
     * \brief Construct from Type[16]
     */
    explicit CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T(Type values[]) :
        column0(values[0], values[1], values[2], values[3]),
        column1(values[4], values[5], values[6], values[7]),
        column2(values[8], values[9], values[10], values[11]),
        column3(values[12], values[13], values[14], values[15])
    {
    }

    /**
     * \brief Construct from a quaternion
     */
    explicit CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T(const CxQuatT<Type>& q)
    {
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

        column0 = CxVec4T<Type>(Type(1.0) - yy - zz, xy + zw, xz - yw, Type(0.0));
        column1 = CxVec4T<Type>(xy - zw, Type(1.0) - xx - zz, yz + xw, Type(0.0));
        column2 = CxVec4T<Type>(xz + yw, yz - xw, Type(1.0) - xx - yy, Type(0.0));
        column3 = CxVec4T<Type>(Type(0.0), Type(0.0), Type(0.0), Type(1.0));
    }

    /**
     * \brief Construct from a diagonal vector
     */
    explicit CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T(const CxVec4T<Type>& diagonal) :
        column0(diagonal.x, Type(0.0), Type(0.0), Type(0.0)),
        column1(Type(0.0), diagonal.y, Type(0.0), Type(0.0)),
        column2(Type(0.0), Type(0.0), diagonal.z, Type(0.0)),
        column3(Type(0.0), Type(0.0), Type(0.0), diagonal.w)
    {
    }

    /**
     * \brief Construct from Mat33 and a translation
     */
    CX_CUDA_CALLABLE CxMat44T(const CxMat33T<Type>& axes, const CxVec3T<Type>& position) :
        column0(axes.column0, Type(0.0)),
        column1(axes.column1, Type(0.0)),
        column2(axes.column2, Type(0.0)),
        column3(position, Type(1.0))
    {
    }

    /**
     * \brief Copy constructor
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T(const CxMat44T& other) :
        column0(other.column0),
        column1(other.column1),
        column2(other.column2),
        column3(other.column3)
    {
    }

    /**
     * \brief Assignment operator
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T& operator=(const CxMat44T& other)
    {
        column0 = other.column0;
        column1 = other.column1;
        column2 = other.column2;
        column3 = other.column3;
        return *this;
    }

    /**
     * \brief Returns true if the two matrices are exactly equal
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE bool operator==(const CxMat44T& m) const
    {
        return column0 == m.column0 && column1 == m.column1 && 
               column2 == m.column2 && column3 == m.column3;
    }

    /**
     * \brief Returns true if the two matrices are not equal
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE bool operator!=(const CxMat44T& m) const
    {
        return !(*this == m);
    }

    /**
     * \brief Get transposed matrix
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T GetTranspose() const
    {
        return CxMat44T(
            CxVec4T<Type>(column0.x, column1.x, column2.x, column3.x),
            CxVec4T<Type>(column0.y, column1.y, column2.y, column3.y),
            CxVec4T<Type>(column0.z, column1.z, column2.z, column3.z),
            CxVec4T<Type>(column0.w, column1.w, column2.w, column3.w));
    }

    /**
     * \brief Unary minus
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T operator-() const
    {
        return CxMat44T(-column0, -column1, -column2, -column3);
    }

    /**
     * \brief Matrix addition
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T operator+(const CxMat44T& other) const
    {
        return CxMat44T(column0 + other.column0, column1 + other.column1, 
                        column2 + other.column2, column3 + other.column3);
    }

    /**
     * \brief Matrix subtraction
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T operator-(const CxMat44T& other) const
    {
        return CxMat44T(column0 - other.column0, column1 - other.column1, 
                        column2 - other.column2, column3 - other.column3);
    }

    /**
     * \brief Scalar multiplication
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T operator*(Type scalar) const
    {
        return CxMat44T(column0 * scalar, column1 * scalar, 
                        column2 * scalar, column3 * scalar);
    }

    /**
     * \brief Matrix multiplication
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T operator*(const CxMat44T& other) const
    {
        // Rows from this <dot> columns from other
        return CxMat44T(Transform(other.column0), Transform(other.column1), 
                        Transform(other.column2), Transform(other.column3));
    }

    /**
     * \brief Equals-add
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T& operator+=(const CxMat44T& other)
    {
        column0 += other.column0;
        column1 += other.column1;
        column2 += other.column2;
        column3 += other.column3;
        return *this;
    }

    /**
     * \brief Equals-subtract
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T& operator-=(const CxMat44T& other)
    {
        column0 -= other.column0;
        column1 -= other.column1;
        column2 -= other.column2;
        column3 -= other.column3;
        return *this;
    }

    /**
     * \brief Equals scalar multiplication
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T& operator*=(Type scalar)
    {
        column0 *= scalar;
        column1 *= scalar;
        column2 *= scalar;
        column3 *= scalar;
        return *this;
    }

    /**
     * \brief Equals matrix multiplication
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T& operator*=(const CxMat44T& other)
    {
        *this = *this * other;
        return *this;
    }

    /**
     * \brief Element access, mathematical way [row, col]
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE Type operator()(CxU32 row, CxU32 col) const
    {
        return (*this)[col][row];
    }

    /**
     * \brief Element access, mathematical way [row, col]
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE Type& operator()(CxU32 row, CxU32 col)
    {
        return (*this)[col][row];
    }

    /**
     * \brief Column access
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec4T<Type>& operator[](CxU32 num)
    {
        CX_ASSERT(num < 4);
        return (&column0)[num];
    }

    /**
     * \brief Column access (const)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE const CxVec4T<Type>& operator[](CxU32 num) const
    {
        CX_ASSERT(num < 4);
        return (&column0)[num];
    }

    /**
     * \brief Transform vector by matrix, equal to v' = M*v
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec4T<Type> Transform(const CxVec4T<Type>& other) const
    {
        return column0 * other.x + column1 * other.y + column2 * other.z + column3 * other.w;
    }

    /**
     * \brief Transform vector by matrix, equal to v' = M*v
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> Transform(const CxVec3T<Type>& other) const
    {
        return Transform(CxVec4T<Type>(other, Type(1.0))).getXYZ();
    }

    /**
     * \brief Rotate vector by matrix, equal to v' = M*v (ignoring translation)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec4T<Type> Rotate(const CxVec4T<Type>& other) const
    {
        return column0 * other.x + column1 * other.y + column2 * other.z;
    }

    /**
     * \brief Rotate vector by matrix, equal to v' = M*v (ignoring translation)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> Rotate(const CxVec3T<Type>& other) const
    {
        return Rotate(CxVec4T<Type>(other, Type(1.0))).getXYZ();
    }

    /**
     * \brief Get basis vector (0=x, 1=y, 2=z)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> GetBasis(CxU32 num) const
    {
        CX_ASSERT(num < 3);
        return (&column0)[num].getXYZ();
    }

    /**
     * \brief Get position (translation) from the matrix
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxVec3T<Type> GetPosition() const
    {
        return column3.getXYZ();
    }

    /**
     * \brief Set position (translation) in the matrix
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE void SetPosition(const CxVec3T<Type>& position)
    {
        column3.x = position.x;
        column3.y = position.y;
        column3.z = position.z;
    }

    /**
     * \brief Get pointer to the first element
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE const Type* Front() const
    {
        return &column0.x;
    }

    /**
     * \brief Get pointer to the first element
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE Type* Front()
    {
        return &column0.x;
    }

    /**
     * \brief Scale columns by a vector
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE void Scale(const CxVec4T<Type>& p)
    {
        column0 *= p.x;
        column1 *= p.y;
        column2 *= p.z;
        column3 *= p.w;
    }

    /**
     * \brief Inverse for rotation-translation matrix
     * Fast inverse assuming the matrix is a rigid transformation (rotation + translation only)
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE CxMat44T InverseRT() const
    {
        const CxVec3T<Type> r0(column0.x, column1.x, column2.x);
        const CxVec3T<Type> r1(column0.y, column1.y, column2.y);
        const CxVec3T<Type> r2(column0.z, column1.z, column2.z);

        return CxMat44T(r0, r1, r2, -(r0 * column3.x + r1 * column3.y + r2 * column3.z));
    }

    /**
     * \brief Returns true if all elements are finite
     */
    CX_CUDA_CALLABLE CX_FORCE_INLINE bool IsFinite() const
    {
        return column0.isFinite() && column1.isFinite() && 
               column2.isFinite() && column3.isFinite();
    }

    /**
     * \brief Get the identity matrix
     */
    CX_CUDA_CALLABLE static CX_FORCE_INLINE CxMat44T Identity()
    {
        return CxMat44T(CxIdentity);
    }

    /**
     * \brief Get a zero matrix
     */
    CX_CUDA_CALLABLE static CX_FORCE_INLINE CxMat44T Zero()
    {
        return CxMat44T(CxZero);
    }
};

/**
 * \brief Scalar multiplication (scalar * matrix)
 */
template<class Type>
CX_CUDA_CALLABLE static CX_FORCE_INLINE CxMat44T<Type> operator*(Type scalar, const CxMat44T<Type>& m)
{
    return m * scalar;
}

// Type definitions
typedef CxMat44T<float> CxMat44;
typedef CxMat44T<double> CxMat44d;

} // namespace CX_NAMESPACE

// Include transform.h to resolve forward declaration
#include "transform.h"

namespace CX_NAMESPACE
{

// Implementation of CxTransformT constructor from CxMat44T
template<class Type>
CX_CUDA_CALLABLE CX_FORCE_INLINE CxTransformT<Type>::CxTransformT(const CxMat44T<Type>& m)
{
    const CxVec3T<Type> column0(m.column0.x, m.column0.y, m.column0.z);
    const CxVec3T<Type> column1(m.column1.x, m.column1.y, m.column1.z);
    const CxVec3T<Type> column2(m.column2.x, m.column2.y, m.column2.z);

    q = CxQuatT<Type>(CxMat33T<Type>(column0, column1, column2));
    p = CxVec3T<Type>(m.column3.x, m.column3.y, m.column3.z);
}

} // namespace CX_NAMESPACE

#endif // MAT44_H
