#ifndef MATH_H
#define MATH_H


#include "macros.h"


#if CX_VC
#pragma warning(push)
#pragma warning(disable : 4985) // 'symbol name': attributes not present on previous declaration
#endif
#include <math.h>

#if CX_VC
#pragma warning(pop)
#endif

#if (CX_LINUX_FAMILY && !CX_ARM_FAMILY)
// Force linking against nothing newer than glibc v2.17 to remain compatible with platforms with older glibc versions
__asm__(".symver expf,expf@GLIBC_2.2.5");
__asm__(".symver powf,powf@GLIBC_2.2.5");
#endif

#include <float.h>
#include "mathIntrinsics.h"



namespace CX_NAMESPACE
{

// constants
static constexpr float CxPi = float(3.141592653589793);
static constexpr float CxHalfPi = float(1.57079632679489661923);
static constexpr float CxTwoPi = float(6.28318530717958647692);
static constexpr float CxInvPi = float(0.31830988618379067154);
static constexpr float CxInvTwoPi = float(0.15915494309189533577);
static constexpr float CxPiDivTwo = float(1.57079632679489661923);
static constexpr float CxPiDivFour = float(0.78539816339744830962);
static constexpr float CxSqrt2 = float(1.4142135623730951);
static constexpr float CxInvSqrt2 = float(0.7071067811865476);

/**
\brief The return value is the greater of the two specified values.
*/
template <class T>
CX_CUDA_CALLABLE CX_FORCE_INLINE T CxMax(T a, T b)
{
	return a < b ? b : a;
}

//! overload for float to use fsel on xbox
template <>
CX_CUDA_CALLABLE CX_FORCE_INLINE float CxMax(float a, float b)
{
	return intrinsics::selectMax(a, b);
}

/**
\brief The return value is the lesser of the two specified values.
*/
template <class T>
CX_CUDA_CALLABLE CX_FORCE_INLINE T CxMin(T a, T b)
{
	return a < b ? a : b;
}

template <>
//! overload for float to use fsel on xbox
CX_CUDA_CALLABLE CX_FORCE_INLINE float CxMin(float a, float b)
{
	return intrinsics::selectMin(a, b);
}

/*
Many of these are just implemented as CX_CUDA_CALLABLE CX_FORCE_INLINE calls to the C lib right now,
but later we could replace some of them with some approximations or more
clever stuff.
*/

/**
\brief abs returns the absolute value of its argument.
*/
CX_CUDA_CALLABLE CX_FORCE_INLINE float CxAbs(float a)
{
	return intrinsics::abs(a);
}

CX_CUDA_CALLABLE CX_FORCE_INLINE bool CxEquals(float a, float b, float eps)
{
	return (CxAbs(a - b) < eps);
}

/**
\brief abs returns the absolute value of its argument.
*/
CX_CUDA_CALLABLE CX_FORCE_INLINE double CxAbs(double a)
{
	return ::fabs(a);
}

/**
\brief abs returns the absolute value of its argument.
*/
CX_CUDA_CALLABLE CX_FORCE_INLINE int32_t CxAbs(int32_t a)
{
	return ::abs(a);
}

/**
\brief Clamps v to the range [hi,lo]
*/
template <class T>
CX_CUDA_CALLABLE CX_FORCE_INLINE T CxClamp(T v, T lo, T hi)
{
	CX_ASSERT(lo <= hi);
	return CxMin(hi, CxMax(lo, v));
}

//!	\brief Square root.
CX_CUDA_CALLABLE CX_FORCE_INLINE float CxSqrt(float a)
{
	return intrinsics::sqrt(a);
}

//!	\brief Square root.
CX_CUDA_CALLABLE CX_FORCE_INLINE double CxSqrt(double a)
{
	return ::sqrt(a);
}

//!	\brief reciprocal square root.
CX_CUDA_CALLABLE CX_FORCE_INLINE float CxRecipSqrt(float a)
{
	return intrinsics::recipSqrt(a);
}

//!	\brief reciprocal square root.
CX_CUDA_CALLABLE CX_FORCE_INLINE double CxRecipSqrt(double a)
{
	return 1 / ::sqrt(a);
}

//!	\brief square of the argument
CX_CUDA_CALLABLE CX_FORCE_INLINE CxF32 CxSqr(const CxF32 a)
{
	return a * a;
}

//! trigonometry -- all angles are in radians.

//!	\brief Sine of an angle ( <b>Unit:</b> Radians )
CX_CUDA_CALLABLE CX_FORCE_INLINE float CxSin(float a)
{
	return intrinsics::sin(a);
}

//!	\brief Sine of an angle ( <b>Unit:</b> Radians )
CX_CUDA_CALLABLE CX_FORCE_INLINE double CxSin(double a)
{
	return ::sin(a);
}

//!	\brief Cosine of an angle (<b>Unit:</b> Radians)
CX_CUDA_CALLABLE CX_FORCE_INLINE float CxCos(float a)
{
	return intrinsics::cos(a);
}

//!	\brief Cosine of an angle (<b>Unit:</b> Radians)
CX_CUDA_CALLABLE CX_FORCE_INLINE double CxCos(double a)
{
	return ::cos(a);
}

//! \brief compute sine and cosine at the same time
CX_CUDA_CALLABLE CX_FORCE_INLINE void CxSinCos(const CxF32 a, CxF32& sin, CxF32& cos)
{
//#if CX_CUDA_COMPILER && __CUDA_ARCH__ >= 350
//	__sincosf(a, &sin, &cos);
//#else
	sin = CxSin(a);
	cos = CxCos(a);
//#endif
}

//! \brief compute sine and cosine at the same time
CX_CUDA_CALLABLE CX_FORCE_INLINE void CxSinCos(const double a, double& sin, double& cos)
{
	sin = CxSin(a);
	cos = CxCos(a);
}

/**
\brief Tangent of an angle.
<b>Unit:</b> Radians
*/
CX_CUDA_CALLABLE CX_FORCE_INLINE float CxTan(float a)
{
	return ::tanf(a);
}

/**
\brief Tangent of an angle.
<b>Unit:</b> Radians
*/
CX_CUDA_CALLABLE CX_FORCE_INLINE double CxTan(double a)
{
	return ::tan(a);
}

/**
\brief Arcsine.
Returns angle between -PI/2 and PI/2 in radians
<b>Unit:</b> Radians
*/
CX_CUDA_CALLABLE CX_FORCE_INLINE float CxAsin(float f)
{
	return ::asinf(CxClamp(f, -1.0f, 1.0f));
}

/**
\brief Arcsine.
Returns angle between -PI/2 and PI/2 in radians
<b>Unit:</b> Radians
*/
CX_CUDA_CALLABLE CX_FORCE_INLINE double CxAsin(double f)
{
	return ::asin(CxClamp(f, -1.0, 1.0));
}

/**
\brief Arccosine.
Returns angle between 0 and PI in radians
<b>Unit:</b> Radians
*/
CX_CUDA_CALLABLE CX_FORCE_INLINE float CxAcos(float f)
{
	return ::acosf(CxClamp(f, -1.0f, 1.0f));
}

/**
\brief Arccosine.
Returns angle between 0 and PI in radians
<b>Unit:</b> Radians
*/
CX_CUDA_CALLABLE CX_FORCE_INLINE double CxAcos(double f)
{
	return ::acos(CxClamp(f, -1.0, 1.0));
}

/**
\brief ArcTangent.
Returns angle between -PI/2 and PI/2 in radians
<b>Unit:</b> Radians
*/
CX_CUDA_CALLABLE CX_FORCE_INLINE float CxAtan(float a)
{
	return ::atanf(a);
}

/**
\brief ArcTangent.
Returns angle between -PI/2 and PI/2 in radians
<b>Unit:</b> Radians
*/
CX_CUDA_CALLABLE CX_FORCE_INLINE double CxAtan(double a)
{
	return ::atan(a);
}

/**
\brief Arctangent of (x/y) with correct sign.
Returns angle between -PI and PI in radians
<b>Unit:</b> Radians
*/
CX_CUDA_CALLABLE CX_FORCE_INLINE float CxAtan2(float x, float y)
{
	return ::atan2f(x, y);
}

/**
\brief Arctangent of (x/y) with correct sign.
Returns angle between -PI and PI in radians
<b>Unit:</b> Radians
*/
CX_CUDA_CALLABLE CX_FORCE_INLINE double CxAtan2(double x, double y)
{
	return ::atan2(x, y);
}

/**
\brief Converts degrees to radians.
*/
CX_CUDA_CALLABLE CX_FORCE_INLINE CxF32 CxDegToRad(const CxF32 a)
{
	return 0.01745329251994329547f * a;
}

//!	\brief returns true if the passed number is a finite floating point number as opposed to INF, NAN, etc.
CX_CUDA_CALLABLE CX_FORCE_INLINE bool CxIsFinite(float f)
{
	return intrinsics::isFinite(f);
}

//!	\brief returns true if the passed number is a finite floating point number as opposed to INF, NAN, etc.
CX_CUDA_CALLABLE CX_FORCE_INLINE bool CxIsFinite(double f)
{
	return intrinsics::isFinite(f);
}

CX_CUDA_CALLABLE CX_FORCE_INLINE float CxFloor(float a)
{
	return ::floorf(a);
}

CX_CUDA_CALLABLE CX_FORCE_INLINE float CxExp(float a)
{
	return ::expf(a);
}

CX_CUDA_CALLABLE CX_FORCE_INLINE float CxCeil(float a)
{
	return ::ceilf(a);
}

CX_CUDA_CALLABLE CX_FORCE_INLINE float CxSign(float a)
{
	return cx::intrinsics::sign(a);
}

CX_CUDA_CALLABLE CX_FORCE_INLINE float CxSign2(float a, float eps = FLT_EPSILON)
{
	return (a < -eps) ? -1.0f : (a > eps) ? 1.0f : 0.0f;
}

CX_CUDA_CALLABLE CX_FORCE_INLINE float CxPow(float x, float y)
{
	return ::powf(x, y);
}

CX_CUDA_CALLABLE CX_FORCE_INLINE float CxLog(float x)
{
	return ::logf(x);
}


} // namespace chysx


#endif