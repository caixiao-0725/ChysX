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
    public:
        ABDModel() {};
        ~ABDModel() {};

        void initializeCPU();

		void calculateAABB();
		void calculateBarycentricCoords();
        void initializeGPU();

    private:
        CxVec3T<CxReal> m_transform;

        ABDDofs m_x;
        DeviceHostVector<CxVec4T<CxReal>> m_barycentric_coords;

        CxAABBT<CxReal> m_box;

        CxReal m_density;
    };
}
#endif