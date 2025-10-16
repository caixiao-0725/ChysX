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

void SingleTriangleMesh::SetName(const std::string& name){
    this->name = name;
}

bool SingleTriangleMesh::ReadFromObjFile(const std::string& filename){
    std::ifstream fin(filename.c_str());
	if (!fin.is_open())
		return false;

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> groupMaterials;

	std::string err;
	std::string base_dir = get_asset_path() + filename;
	if (base_dir.empty())
	{
		base_dir = ".";
	}

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
	if ((int)(attrib.texcoords.size()) / 3 > 0)
		hasTextureCoords = true;
	if ((int)groupMaterials.size() > 0)
		hasMaterials = true;

	int pointsNum = (int)(attrib.vertices.size()) / 3;
	int textureCoordNum = (int)(attrib.texcoords.size()) / 2;

    vertices.AllocateHost(pointsNum);
}