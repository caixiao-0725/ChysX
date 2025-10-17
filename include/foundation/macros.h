//author  :  ChengXiao (vegtsunami@gmail.com)

#pragma once

#include <cstdio>
#include <assert.h>

#pragma warning(disable : 4201)  //!	Nonstandard extension used: nameless struct/union.
/*********************************************************************************
*******************************    C++ Version     *******************************
*********************************************************************************/

#if defined(_MSVC_LANG)
	#define CX_CPLUSPLUS			_MSVC_LANG
#else
	#define CX_CPLUSPLUS			__cplusplus
#endif

#if CX_CPLUSPLUS > 202002L
	#define CX_CPP_VERSION 23
#elif CX_CPLUSPLUS > 201703L
	#define CX_CPP_VERSION 20
#elif CX_CPLUSPLUS > 201402L
	#define CX_CPP_VERSION 17
#else
	#error "Requires at least c++ standard version 17"
#endif

#define CX_HAS_CXX_20				(CX_CPP_VERSION >= 20)
#define CX_HAS_CXX_17				(CX_CPP_VERSION >= 17)


/*********************************************************************************
****************************    Operating System     *****************************
*********************************************************************************/

/**
Operating system defines, see http://sourceforge.net/p/predef/wiki/OperatingSystems/
*/
#if defined(_WIN64)
	#define CX_WIN64 1
#elif defined(_WIN32) // note: _M_PPC implies _WIN32
	#define CX_WIN32 1
#elif defined(__linux__) || defined (__EMSCRIPTEN__)
	#define CX_LINUX 1
#elif defined(__APPLE__)
	#define CX_OSX 1
#elif defined(__NX__)
	#define CX_SWITCH 1
#else
	#error "Unknown operating system"
#endif

#if defined(_MSC_VER)
#if _MSC_VER >= 1920
	#define CX_VC 16
#elif _MSC_VER >= 1910
	#define CX_VC 15
#elif _MSC_VER >= 1900
	#define CX_VC 14
#elif _MSC_VER >= 1800
	#define CX_VC 12
#elif _MSC_VER >= 1700
	#define CX_VC 11
#elif _MSC_VER >= 1600
	#define CX_VC 10
#elif _MSC_VER >= 1500
	#define CX_VC 9
#else
	#error "Unknown VC version"
#endif
#elif defined(__clang__)
#define CX_CLANG 1
	#if defined (__clang_major__) 
		#define CX_CLANG_MAJOR __clang_major__
	#elif defined (_clang_major)
		#define CX_CLANG_MAJOR _clang_major
	#else
		#define CX_CLANG_MAJOR 0
	#endif	
#elif defined(__GNUC__) // note: __clang__ implies __GNUC__
	#define CX_GCC 1
#else
	#error "Unknown compiler"
#endif

/**
define anything not defined on this platform to 0
*/
#ifndef CX_VC
	#define CX_VC 0
#endif
#ifndef CX_CLANG
	#define CX_CLANG 0
#endif
#ifndef CX_GCC
	#define CX_GCC 0
#endif
#ifndef CX_WIN64
	#define CX_WIN64 0
#endif
#ifndef CX_WIN32
	#define CX_WIN32 0
#endif
#ifndef CX_LINUX
	#define CX_LINUX 0
#endif
#ifndef CX_OSX
	#define CX_OSX 0
#endif
#ifndef CX_SWITCH
	#define CX_SWITCH 0
#endif
#ifndef CX_X64
	#define CX_X64 0
#endif
#ifndef CX_X86
	#define CX_X86 0
#endif
#ifndef CX_A64
	#define CX_A64 0
#endif
#ifndef CX_ARM
	#define CX_ARM 0
#endif
#ifndef CX_PPC
	#define CX_PPC 0
#endif
#ifndef CX_SSE2
	#define CX_SSE2 0
#endif
#ifndef CX_NEON
	#define CX_NEON 0
#endif
#ifndef CX_VMX
	#define CX_VMX 0
#endif

/**
family shortcuts
*/
// compiler
#define CX_GCC_FAMILY (CX_CLANG || CX_GCC)
// os
#define CX_WINDOWS_FAMILY (CX_WIN32 || CX_WIN64)
#define CX_LINUX_FAMILY CX_LINUX
#define CX_APPLE_FAMILY CX_OSX                              // equivalent to #if __APPLE__
#define CX_UNIX_FAMILY (CX_LINUX_FAMILY || CX_APPLE_FAMILY) // shortcut for unix/posix platforms
#if defined(__EMSCRIPTEN__)
	#define CX_EMSCRIPTEN 1
#else
	#define CX_EMSCRIPTEN 0
#endif
// architecture
#define CX_INTEL_FAMILY (CX_X64 || CX_X86)
#define CX_ARM_FAMILY (CX_ARM || CX_A64)
#define CX_P64_FAMILY (CX_X64 || CX_A64) // shortcut for 64-bit architectures

#if CX_LINUX && CX_CLANG && !CX_CUDA_COMPILER
#define CX_COMPILE_TIME_ASSERT(exp) \
_Pragma(" clang diagnostic push") \
_Pragma(" clang diagnostic ignored \"-Wc++98-compat\"") \
static_assert(exp, "") \
_Pragma(" clang diagnostic pop")
#else
#define CX_COMPILE_TIME_ASSERT(exp) static_assert(exp, "")
#endif


/*********************************************************************************
****************************    Compiling Numbers     ****************************
*********************************************************************************/

#define CX_MIN(a, b)				(((a) < (b)) ? (a) : (b))
#define CX_MAX(a, b)				(((a) > (b)) ? (a) : (b))

/*********************************************************************************
***************************    CUDA Compatibilities    ***************************
*********************************************************************************/

#if defined(__CUDACC__)
	#define	CX_INLINE						__inline__
	#define CX_ALIGN(n)						__align__(n)
	#define CX_FORCE_INLINE					__forceinline__
	#define	CX_CUDA_CALLABLE				__host__ __device__
	#define	CX_CUDA_CALLABLE_INLINE			__host__ __device__ __inline__
#else
	#define	CX_INLINE						inline
	#define CX_ALIGN(n)						alignas(n)
	#define CX_FORCE_INLINE					__forceinline
	#define	CX_CUDA_CALLABLE
	#define	CX_CUDA_CALLABLE_INLINE			inline

	#ifndef __device__
		#define __device__
	#endif
#endif

#if defined(__CUDA_ARCH__)
	#define	CX_ASSERT(expression)
#else
	#define	CX_ASSERT(expression)			assert(expression)
#endif

#if defined(__CUDACC__)
#define CX_CUDA_COMPILER 1
#else
#define CX_CUDA_COMPILER 0
#endif

/*********************************************************************************
**********************************    Utils    ***********************************
*********************************************************************************/

#define CX_NODISCARD						_NODISCARD
#define CX_NOVTABLE							__declspec(novtable)

#if defined(DEBUG) || defined(_DEBUG)
	#define CX_DEBUG
#endif

/*
define anything not defined through the command line to 0
*/
#ifndef CX_DEBUG
	#define CX_DEBUG 0
#endif
#ifndef CX_CHECKED
	#define CX_CHECKED 0
#endif
#ifndef CX_PROFILE
	#define CX_PROFILE 0
#endif
#ifndef CX_DEBUG_CRT
	#define CX_DEBUG_CRT 0
#endif
#ifndef CX_NVTX
	#define CX_NVTX 0
#endif
#ifndef CX_DOXYGEN
	#define CX_DOXYGEN 0
#endif

/*********************************************************************************
*******************************    Noncopyable    ********************************
*********************************************************************************/

#define CX_NONCOPYABLE(ClassName)												\
																				\
	ClassName(const ClassName&) = delete;										\
																				\
	ClassName & operator=(const ClassName&) = delete;							\

/*********************************************************************************
*******************************    Movable    ********************************
*********************************************************************************/

#define CX_MOVABLE(ClassName)												    \
																				\
	ClassName(ClassName&&) noexcept = default;									\
																				\
	ClassName & operator=(ClassName&&) noexcept = default;						\

/*********************************************************************************
********************************    Namespace    *********************************
*********************************************************************************/

#ifndef CX_NAMESPACE
	#define CX_NAMESPACE				cx
#endif

#define CX_USING_NAMESPACE				using namespace CX_NAMESPACE;

namespace CX_NAMESPACE { namespace dev {} }
namespace dev = CX_NAMESPACE::dev;