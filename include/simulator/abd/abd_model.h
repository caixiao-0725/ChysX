#ifndef ABD_MODEL_H
#define ABD_MODEL_H

#include "geometry/triangle_mesh.h"
#include "abd_utils.h"
#include "foundation/aabb.h"
#include "device_host_vector.h"
#include "foundation/vec4.h"

namespace CX_NAMESPACE
{
    class ABDModel : public SingleTriangleMesh
    {
        CX_NONCOPYABLE(ABDModel)
    public:
        ABDModel() {};
        ABDModel(const CxVec3T<CxReal>& center, const CxVec3T<CxReal>& size):m_transform(center), m_size(size) {};
        ~ABDModel() {};
        // Enable move semantics
        CX_MOVABLE(ABDModel)

        void initializeCPU();

		void calculateAABB();
		void calculateBarycentricCoords();
        void initializeGPU();

    private:
        CxVec3T<CxReal> m_transform = CxVec3T<CxReal>(0,0,0);
        CxVec3T<CxReal> m_size = CxVec3T<CxReal>(1,1,1);

        ABDDofs m_x;
        ABDDofs m_x_rest;
        DeviceHostVector<CxVec4T<CxReal>> m_barycentric_coords;

        CxAABBT<CxReal> m_box;

        CxReal m_density;
    };
}
#endif