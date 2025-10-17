#ifndef ABD_UTILS_H
#define ABD_UTILS_H

#include "foundation/vec3.h"
#include "foundation/vec4.h"
#include "foundation/mat33.h"
#include <vector>


namespace CX_NAMESPACE
{
    struct ABDDofs {
        CxVec3T<CxReal> v[4];
    };

    CxReal computeElementSpace(const CxVec3T<CxReal>& v0, const CxVec3T<CxReal>& v1, const CxVec3T<CxReal>& v2,const CxVec3T<CxReal>& v3); // area or volume

	void computeBarycentricCoordinate(const CxVec3T<CxReal>& v0, const CxVec3T<CxReal>& v1, const CxVec3T<CxReal>& v2,const CxVec3T<CxReal>& v3,
        const std::vector<CxVec3T<CxReal>>& target, std::vector<CxVec4T<CxReal>>& baryCoord);

}

#endif // ABD_UTILS_H