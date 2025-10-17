#pragma once

#include <string>
#include "foundation/vec2.h"
#include "foundation/vec3.h"
#include "device_host_vector.h"


namespace CX_NAMESPACE
{
    class SingleTriangleMesh
    {
        CX_NONCOPYABLE(SingleTriangleMesh)
    public:
        SingleTriangleMesh();
        ~SingleTriangleMesh();
        // Enable move semantics so containers can reallocate by moving
        CX_MOVABLE(SingleTriangleMesh)

        void SetName(const std::string& name);
        std::string GetName() const { return name; }

        bool ReadFromObjFile(const std::string& filename);
        
        // Getter methods for testing and access
        size_t GetSize() const { return pointsNum; }
        size_t GetFaceCount() const { return facesNum; }
        bool HasNormals() const { return hasNormals; }
        bool HasTextureCoords() const { return hasTextureCoords; }
        
        const DeviceHostVector<CxVec3>& GetVertices() const { return vertices; }
        const DeviceHostVector<CxU32>& GetIndices() const { return indices; }
        const DeviceHostVector<CxVec3>& GetNormals() const { return normals; }
        const DeviceHostVector<CxVec2>& GetTextureCoords() const { return texcoords; }
        
        // Non-const versions for accessing host data
        DeviceHostVector<CxVec3>& GetVertices() { return vertices; }
        DeviceHostVector<CxU32>& GetIndices() { return indices; }
        DeviceHostVector<CxVec3>& GetNormals() { return normals; }
        DeviceHostVector<CxVec2>& GetTextureCoords() { return texcoords; }

    private:
        std::string name;
        bool hasNormals;
        bool hasTextureCoords;  

        int pointsNum; 
        int facesNum;

        DeviceHostVector<CxVec3> vertices;
        DeviceHostVector<CxU32> indices;
		DeviceHostVector<CxVec3> normals;
        DeviceHostVector<CxVec2> texcoords;
    };
}