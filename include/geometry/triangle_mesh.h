#pragma once

#include <string>
#include "foundation/vec3.h"
#include "device_host_vector.h"


namespace CX_NAMESPACE
{
    class SingleTriangleMesh
    {
    public:
        SingleTriangleMesh();
        ~SingleTriangleMesh();

        void SetName(const std::string& name);
        std::string GetName() const { return name; }

        bool ReadFromObjFile(const std::string& filename);

    private:
        std::string name;
        bool hasNormals;
        bool hasTextureCoords;  

        DeviceHostVector<CxVec3> vertices;
        DeviceHostVector<CxU32> indices;
		DeviceHostVector<CxVec3> normals;
    };
}