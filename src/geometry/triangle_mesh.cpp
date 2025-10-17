#include "geometry/triangle_mesh.h"
#include "geometry/tiny_obj_loader.h"
#include "foundation/io_path.h"
#include <iostream>

CX_USING_NAMESPACE

SingleTriangleMesh::SingleTriangleMesh()
{
}

SingleTriangleMesh::~SingleTriangleMesh()
{
}

void SingleTriangleMesh::SetName(const std::string& file_name){
    this->name = file_name;
}

bool SingleTriangleMesh::ReadFromObjFile(const std::string& filename){
    std::ifstream fin(filename.c_str());
	if (!fin.is_open())
		return false;

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> groupMaterials;

	std::string err;
	//std::string base_dir = get_asset_path() + filename;
	//if (base_dir.empty())
	//{
    std::string	base_dir = ".";
	//}

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &groupMaterials, &err, filename.c_str(), base_dir.c_str(), true);
	if (!err.empty())
	{
		std::cerr << err << std::endl;
	}
	if (!ret) {
		std::cerr << "Error: Failed to load " << filename << " !" << std::endl;
		fin.close();
		return false;
	}

    hasNormals = false;
	hasTextureCoords = false;
	bool hasMaterials = false;

	if ((int)(attrib.normals.size()) / 3 > 0)
		hasNormals = true;
	if ((int)(attrib.texcoords.size()) / 2 > 0)
		hasTextureCoords = true;
	if ((int)groupMaterials.size() > 0)
		hasMaterials = true;

	pointsNum = (int)(attrib.vertices.size()) / 3;
	int textureCoordNum = (int)(attrib.texcoords.size()) / 2;

    // Allocate memory for vertices
    vertices.AllocateHost(pointsNum);
    
    // Process vertices
    for (int i = 0; i < pointsNum; i++)
    {
        CxVec3 vertex(attrib.vertices[3 * i], attrib.vertices[3 * i + 1], attrib.vertices[3 * i + 2]);
        vertices.GetHost()[i] = vertex;
    }

    // Process faces and indices
    facesNum = 0;
    std::vector<CxU32> tempIndices;
    
    for (size_t s = 0; s < shapes.size(); s++)
    {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
        {
            int fv = shapes[s].mesh.num_face_vertices[f];
            if (fv == 3) // Only process triangular faces
            {
                for (int v = 0; v < fv; v++)
                {
                    tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                    tempIndices.push_back(idx.vertex_index);
                }
                facesNum++;
            }
            index_offset += fv;
        }
    }
    
    // Allocate and store indices
    indices.AllocateHost(tempIndices.size());
    for (size_t i = 0; i < tempIndices.size(); i++)
    {
        indices.GetHost()[i] = tempIndices[i];
    }

    // Process normals if available
    if (hasNormals)
    {
        normals.AllocateHost(pointsNum);
        for (int i = 0; i < pointsNum; i++)
        {
            CxVec3 normal(attrib.normals[3 * i], attrib.normals[3 * i + 1], attrib.normals[3 * i + 2]);
            normals.GetHost()[i] = normal;
        }
    }

    // Process texture coordinates if available
    if (hasTextureCoords)
    {
        texcoords.AllocateHost(textureCoordNum);
        for (int i = 0; i < textureCoordNum; i++)
        {
            CxVec2 texcoord(attrib.texcoords[2 * i], attrib.texcoords[2 * i + 1]);
            texcoords.GetHost()[i] = texcoord;
        }
    }

    fin.close();
    return true;
}