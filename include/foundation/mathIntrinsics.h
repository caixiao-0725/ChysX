#ifndef CX_MATH_INTRINSICS_H
#define CX_MATH_INTRINSICS_H

#include <string.h>
#include "macros.h"
#include "simpleTypes.h"

#if CX_WINDOWS_FAMILY
#include "windows/windowsMathIntrinsics.h"
#elif(CX_LINUX || CX_APPLE_FAMILY)
#include "unix/PxUnixMathIntrinsics.h"
#elif CX_SWITCH
#include "switch/PxSwitchMathIntrinsics.h"
#else
#error "Platform not supported!"
#endif

/**
Platform specific defines
*/
#if CX_WINDOWS_FAMILY
#pragma intrinsic(abs)
#pragma intrinsic(labs)
#endif

#endif 
