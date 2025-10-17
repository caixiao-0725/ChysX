#ifndef WINDOWS_MATH_INTRINSICS_H
#define WINDOWS_MATH_INTRINSICS_H

#include "foundation/macros.h"

#if !CX_WINDOWS_FAMILY
#error "This file should only be included by Windows builds!!"
#endif

#include <math.h>
#include <float.h>

namespace CX_NAMESPACE
{
namespace intrinsics
{


//! \brief platform-specific absolute value
CX_CUDA_CALLABLE CX_FORCE_INLINE float abs(float a)
{
	return ::fabsf(a);
}

//! \brief platform-specific select float
CX_CUDA_CALLABLE CX_FORCE_INLINE float fsel(float a, float b, float c)
{
	return (a >= 0.0f) ? b : c;
}

//! \brief platform-specific sign
CX_CUDA_CALLABLE CX_FORCE_INLINE float sign(float a)
{
	return (a >= 0.0f) ? 1.0f : -1.0f;
}

//! \brief platform-specific reciprocal
CX_CUDA_CALLABLE CX_FORCE_INLINE float recip(float a)
{
	return 1.0f / a;
}

//! \brief platform-specific reciprocal estimate
CX_CUDA_CALLABLE CX_FORCE_INLINE float recipFast(float a)
{
	return 1.0f / a;
}

//! \brief platform-specific square root
CX_CUDA_CALLABLE CX_FORCE_INLINE float sqrt(float a)
{
	return ::sqrtf(a);
}

//! \brief platform-specific reciprocal square root
CX_CUDA_CALLABLE CX_FORCE_INLINE float recipSqrt(float a)
{
	return 1.0f / ::sqrtf(a);
}

//! \brief platform-specific reciprocal square root estimate
CX_CUDA_CALLABLE CX_FORCE_INLINE float recipSqrtFast(float a)
{
	return 1.0f / ::sqrtf(a);
}

//! \brief platform-specific sine
CX_CUDA_CALLABLE CX_FORCE_INLINE float sin(float a)
{
	return ::sinf(a);
}

//! \brief platform-specific cosine
CX_CUDA_CALLABLE CX_FORCE_INLINE float cos(float a)
{
	return ::cosf(a);
}

//! \brief platform-specific minimum
CX_CUDA_CALLABLE CX_FORCE_INLINE float selectMin(float a, float b)
{
	return a < b ? a : b;
}

//! \brief platform-specific maximum
CX_CUDA_CALLABLE CX_FORCE_INLINE float selectMax(float a, float b)
{
	return a > b ? a : b;
}

//! \brief platform-specific finiteness check (not INF or NAN)
CX_CUDA_CALLABLE CX_FORCE_INLINE bool isFinite(float a)
{
#if CX_CUDA_COMPILER
	return !!isfinite(a);
#else
	return (0 == ((_FPCLASS_SNAN | _FPCLASS_QNAN | _FPCLASS_NINF | _FPCLASS_PINF) & _fpclass(a)));
#endif
}

//! \brief platform-specific finiteness check (not INF or NAN)
CX_CUDA_CALLABLE CX_FORCE_INLINE bool isFinite(double a)
{
#if CX_CUDA_COMPILER
	return !!isfinite(a);
#else
	return (0 == ((_FPCLASS_SNAN | _FPCLASS_QNAN | _FPCLASS_NINF | _FPCLASS_PINF) & _fpclass(a)));
#endif
}

/*!
Sets \c count bytes starting at \c dst to zero.
*/
CX_FORCE_INLINE void* memZero(void* dest, size_t count)
{
	return memset(dest, 0, count);
}

/*!
Sets \c count bytes starting at \c dst to \c c.
*/
CX_FORCE_INLINE void* memSet(void* dest, int32_t c, size_t count)
{
	return memset(dest, c, count);
}

/*!
Copies \c count bytes from \c src to \c dst. User memMove if regions overlap.
*/
CX_FORCE_INLINE void* memCopy(void* dest, const void* src, size_t count)
{
	return memcpy(dest, src, count);
}

/*!
Copies \c count bytes from \c src to \c dst. Supports overlapping regions.
*/
CX_FORCE_INLINE void* memMove(void* dest, const void* src, size_t count)
{
	return memmove(dest, src, count);
}

} // namespace intrinsics
} // namespace ChysX

#endif