#ifndef ABD_UTILS_H
#define ABD_UTILS_H

#include "foundation/vec3.h"
#include "foundation/mat33.h"
#include <vector>


namespace CX_NAMESPACE
{
    struct ABDDofs {
        CxVec3T<CxReal> v[4];
    };

    CxReal computeElementSpace(const CxVec3T<CxReal>& v0, const CxVec3T<CxReal>& v1, const CxVec3T<CxReal>& v2,const CxVec3T<CxReal>& v3) // area or volume
	{
		CxReal space = 0;
		CxMat33T<CxReal> D(v1-v0,v2-v0,v3-v0);
		space = CxAbs(D.getDeterminant() / 6);
		return space;
	}

	void computeBarycentricCoordinate(const CxVec3T<CxReal>& v0, const CxVec3T<CxReal>& v1, const CxVec3T<CxReal>& v2,const CxVec3T<CxReal>& v3,
        std::vector<CxVec3T<CxReal>>& target, std::vector<CxVec4T<CxReal>>& baryCoord)
	{
		CxReal vSpace = computeElementSpace(v0,v1,v2,v3);
        baryCoord.resize(target.size());
        for(int i = 0; i < target.size(); i++){
            auto p = target[i]; 
            CxVec4T<CxReal> bc(0,0,0,0);
            CxReal space0 = computeElementSpace(v1,v2,v3,p);
            bc.x = space0 / vSpace;
            CxReal space1 = computeElementSpace(v0,v2,v3,p);
            bc.y = space1 / vSpace;
            CxReal space2 = computeElementSpace(v0,v1,v3,p);
            bc.z = space2 / vSpace;
            CxReal space3 = computeElementSpace(v0,v1,v2,p);
            bc.w = space3 / vSpace;
            baryCoord[i] = bc;
        }
        return ;
	}

}

#endif // ABD_UTILS_H