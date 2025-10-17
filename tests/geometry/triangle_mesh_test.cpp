#include "triangle_mesh.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include "foundation/io_path.h"

CX_USING_NAMESPACE

void triangle_mesh_cube_test()
{
    std::cout << "Testing triangle mesh cube loading..." << std::endl;
    
    SingleTriangleMesh mesh;
    
    // Test loading the cube.obj file
    bool success = mesh.ReadFromObjFile(get_asset_path()+"cube/cube.obj");
    assert(success && "Failed to load cube.obj file");
    
    std::cout << "✓ Successfully loaded cube.obj" << std::endl;
    
    // Verify mesh properties
    // Cube should have 8 vertices
    assert(mesh.GetSize() == 8 && "Cube should have 8 vertices");
    std::cout << "✓ Cube has 8 vertices" << std::endl;
    
    // Cube should have 12 faces (6 faces * 2 triangles each)
    assert(mesh.GetFaceCount() == 12 && "Cube should have 12 triangular faces");
    std::cout << "✓ Cube has 12 triangular faces" << std::endl;
    
    // Verify vertices are loaded correctly
    const auto& vertices = mesh.GetVertices();
    assert(vertices.GetSize() == 8 && "Vertices vector should have 8 elements");
    
    // Check some specific vertices (cube corners)
    const auto& hostVertices = vertices.GetHost();
    
    // Check first vertex (should be -1, -1, -1)
    CxVec3 expectedFirst(-1.0f, -1.0f, -1.0f);
    assert(std::abs(hostVertices[0].x - expectedFirst.x) < 1e-6f);
    assert(std::abs(hostVertices[0].y - expectedFirst.y) < 1e-6f);
    assert(std::abs(hostVertices[0].z - expectedFirst.z) < 1e-6f);
    std::cout << "✓ First vertex is correct: (" << hostVertices[0].x << ", " << hostVertices[0].y << ", " << hostVertices[0].z << ")" << std::endl;
    
    // Check second vertex (should be 1, -1, -1)
    CxVec3 expectedSecond(1.0f, -1.0f, -1.0f);
    assert(std::abs(hostVertices[1].x - expectedSecond.x) < 1e-6f);
    assert(std::abs(hostVertices[1].y - expectedSecond.y) < 1e-6f);
    assert(std::abs(hostVertices[1].z - expectedSecond.z) < 1e-6f);
    std::cout << "✓ Second vertex is correct: (" << hostVertices[1].x << ", " << hostVertices[1].y << ", " << hostVertices[1].z << ")" << std::endl;
    
    // Verify indices are loaded correctly
    const auto& indices = mesh.GetIndices();
    assert(indices.GetSize() == 36 && "Indices should have 36 elements (12 faces * 3 vertices each)");
    
    const auto& hostIndices = indices.GetHost();
    
    // Check first triangle indices (should be 0, 1, 2 for first face)
    assert(hostIndices[0] == 0 && "First triangle first vertex index should be 0");
    assert(hostIndices[1] == 1 && "First triangle second vertex index should be 1");
    assert(hostIndices[2] == 2 && "First triangle third vertex index should be 2");
    std::cout << "✓ First triangle indices are correct: [" << hostIndices[0] << ", " << hostIndices[1] << ", " << hostIndices[2] << "]" << std::endl;
    
    // Verify normals are loaded (cube should have normals)
    assert(mesh.HasNormals() && "Cube should have normals");
    const auto& normals = mesh.GetNormals();
    assert(normals.GetSize() == 8 && "Normals should have 8 elements");
    std::cout << "✓ Cube has normals loaded" << std::endl;
    
    // Verify texture coordinates are loaded
    assert(mesh.HasTextureCoords() && "Cube should have texture coordinates");
    const auto& texcoords = mesh.GetTextureCoords();
    assert(texcoords.GetSize() == 4 && "Texture coordinates should have 4 elements");
    std::cout << "✓ Cube has texture coordinates loaded" << std::endl;
    
    // Test mesh name functionality
    mesh.SetName("TestCube");
    assert(mesh.GetName() == "TestCube" && "Mesh name should be set correctly");
    std::cout << "✓ Mesh name functionality works" << std::endl;
    
    std::cout << "✓ All triangle mesh cube tests passed!" << std::endl;
    std::cout << std::endl;
}
