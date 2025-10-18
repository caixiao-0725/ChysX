#include "abd/abd_model.h"
#include "abd/abd_utils.h"
namespace CX_NAMESPACE
{


	void ABDModel::initialize()
	{
		calculateAABB();
		calculateBarycentricCoords();


		auto delta1 = m_x.v[1] - m_x.v[0];
		auto delta2 = m_x.v[2] - m_x.v[0];
		auto delta3 = m_x.v[3] - m_x.v[0];

		m_x_rest.v[0] = CxVec3T<CxReal>(0,0,0);
		m_x_rest.v[1] = CxVec3T<CxReal>(m_size.x*delta1.x,m_size.y*delta1.y,m_size.x*delta1.z);
		m_x_rest.v[2] = CxVec3T<CxReal>(m_size.x*delta2.x,m_size.y*delta2.y,m_size.x*delta2.z);
		m_x_rest.v[3] = CxVec3T<CxReal>(m_size.x*delta3.x,m_size.y*delta3.y,m_size.x*delta3.z);

		m_x.v[0] = m_x_rest.v[0] + m_transform;
		m_x.v[1] = m_x_rest.v[1] + m_transform;
		m_x.v[2] = m_x_rest.v[2] + m_transform;
		m_x.v[3] = m_x_rest.v[3] + m_transform;
	}

	void ABDModel::calculateAABB()
	{
		// Get vertices from the mesh
		const auto& meshVertices = GetVertices();
		size_t vertexCount = GetSize();

		if (vertexCount == 0)
		{
			// Empty mesh, create invalid AABB
			m_box.Empty();
			return;
		}

		// Get host data
		const auto& hostVertices = meshVertices.GetHost();

		// Initialize AABB with first vertex
		m_box = CxAABB(hostVertices[0]);

		// Expand AABB to include all vertices
		for (size_t i = 1; i < vertexCount; i++)
		{
			m_box.Combine(hostVertices[i]);
		}
	}

	void ABDModel::calculateBarycentricCoords(){
		m_x.v[0] = m_box.Corner(0);

		CxReal r = 2 * (m_box._max - m_box._min).magnitude();

		for (int i = 1; i < 4; i++) // global coordinate
		{
			m_x.v[i] = m_x.v[0];
			m_x.v[i][i - 1] += r;
		}

		// Get non-const references for the function call
		auto& hostVertices = GetVertices().GetHost();

		computeBarycentricCoordinate(m_x.v[0], m_x.v[1], m_x.v[2], m_x.v[3], hostVertices, m_barycentric_coords);

	}


}