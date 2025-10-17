#include "abd/abd_model.h"

namespace CX_NAMESPACE
{
	void ABDModel::initializeCPU()
	{
		// Placeholder for CPU initialization logic
	}

	void ABDModel::calculateAABB()
	{
		// Get vertices from the mesh
		const auto& vertices = GetVertices();
		size_t vertexCount = GetSize();

		if (vertexCount == 0)
		{
			// Empty mesh, create invalid AABB
			m_box.Empty();
			return;
		}

		// Get host data
		const auto& hostVertices = vertices.GetHost();

		// Initialize AABB with first vertex
		m_box = CxAABB(hostVertices[0]);

		// Expand AABB to include all vertices
		for (size_t i = 1; i < vertexCount; i++)
		{
			m_box.Combine(hostVertices[i]);
		}
	}

	void ABDModel::calculateBarycentricCoords(){
		// Get vertices from the mesh
		const auto& vertices = GetVertices();
		size_t vertexCount = GetSize();

		m_barycentric_coords.AllocateHost(vertexCount);

		m_x.v[0] = m_box.Corner(0);

		CxReal r = 2 * (m_box.GetMax() - m_box.GetMin()).magnitude();

		for (int i = 1; i < 4; i++) // global coordinate
		{
			m_x.v[i] = m_x.v[0];
			m_x.v[i][i - 1] += r;
		}

		

	}


	void ABDModel::initializeGPU()
	{
		// Placeholder for GPU initialization logic
	}
}