#ifndef CONSTRUCTOR_H
#define CONSTRUCTOR_H

#include "macros.h"

namespace CX_NAMESPACE
{

/** enum for zero constructor tag for vectors and matrices */
enum CxZERO
{
	CxZero
};

/** enum for identity constructor flag for quaternions, transforms, and matrices */
enum CxIDENTITY
{
	CxIdentity
};

} 

#endif //CONSTRUCTOR_H
