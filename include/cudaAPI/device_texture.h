/**
 *	Copyright (c) 2025 Wenchao Huang <physhuangwenchao@gmail.com>
 *
 *	Permission is hereby granted, free of charge, to any person obtaining a copy
 *	of this software and associated documentation files (the "Software"), to deal
 *	in the Software without restriction, including without limitation the rights
 *	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *	copies of the Software, and to permit persons to whom the Software is
 *	furnished to do so, subject to the following conditions:
 *
 *	The above copyright notice and this permission notice shall be included in all
 *	copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *	SOFTWARE.
 */
#pragma once

#include "fwd.h"
#include "format.h"

#ifdef __CUDACC__	//	For CUDA-based project.
	#include <texture_indirect_functions.h>
#endif

namespace CX_NAMESPACE::dev
{
	/*****************************************************************************
	*******************************    Texture    ********************************
	*****************************************************************************/

	struct Texture		//	Base class for all Texture objects.
	{
		//	Default constructor.
		CX_CUDA_CALLABLE Texture() {}

		//	Constructor with nullptr.
		CX_CUDA_CALLABLE Texture(std::nullptr_t) : m_hTexture(0) {}

		//	Constructor with cudaTextureObject_t.
		CX_CUDA_CALLABLE explicit Texture(cudaTextureObject_t hTexture) : m_hTexture(hTexture) {}

		//	Return CUDA type of this object for compatibility.
		CX_CUDA_CALLABLE cudaTextureObject_t handle() const { return m_hTexture; }

		//	Bool conversion operator.
		CX_CUDA_CALLABLE operator bool() const { return m_hTexture != 0; }

		//	Check if the surface is empty.
		CX_CUDA_CALLABLE bool empty() const { return m_hTexture == 0; }

	protected:

		cudaTextureObject_t			m_hTexture;
	};

	/*****************************************************************************
	*****************************    Tex1D<Type>    ******************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D texture object for device access.
	 *	@tparam		Type - The data type readed from the texture.
	 *	@details	This struct provides an interface for 1D texture object.
	 */
	template<typename Type> struct Tex1D : public Texture
	{
		//	The base type without const or volatile qualifiers.
		using value_type = std::remove_cv_t<Type>;

		//	Default constructor.
		CX_CUDA_CALLABLE Tex1D() {}

		//	Constructor with nullptr.
		CX_CUDA_CALLABLE Tex1D(std::nullptr_t) : Texture(nullptr) {}

		//	Constructor with cudaTextureObject_t.
		CX_CUDA_CALLABLE explicit Tex1D(cudaTextureObject_t hTexture) : Texture(hTexture) {}

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type fetch(float x) const;
	#else
		__device__ value_type fetch(float x) const
		{
			InternalValueType<Type> value;

			tex1D<decltype(value)>(&value, m_hTexture, x);

			return reinterpret_cast<value_type&>(value);
		}
	#endif
	};

	/*****************************************************************************
	****************************    Tex1DLod<Type>    ****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D lod texture object for device access.
	 *	@tparam		Type - The data type readed from the texture.
	 *	@details	This struct provides an interface for 1D lod texture object.
	 */
	template<typename Type> struct Tex1DLod : public Texture
	{
		//	The base type without const or volatile qualifiers.
		using value_type = std::remove_cv_t<Type>;

		//	Default constructor.
		CX_CUDA_CALLABLE Tex1DLod() : Texture() {}

		//	Constructor with nullptr.
		CX_CUDA_CALLABLE Tex1DLod(std::nullptr_t) : Texture(nullptr) {}

		//	Constructor with cudaTextureObject_t.
		CX_CUDA_CALLABLE explicit Tex1DLod(cudaTextureObject_t hTexture) : Texture(hTexture) {}

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type fetch(float x, float level) const;
	#else
		__device__ value_type fetch(float x, float level) const
		{
			InternalValueType<Type> value;

			tex1DLod<decltype(value)>(&value, m_hTexture, x, level);

			return reinterpret_cast<value_type&>(value);
		}
	#endif

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type grad(float x, float dx, float dy) const;
	#else
		__device__ value_type grad(float x, float dx, float dy) const
		{
			InternalValueType<Type> value;

			tex1DGrad<decltype(value)>(&value, m_hTexture, x, dx, dy);

			return reinterpret_cast<value_type&>(value);
		}
	#endif
	};

	/*****************************************************************************
	**************************    Tex1DLayered<Type>    **************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D layered CUDA texture object for device access.
	 *	@tparam		Type - The data type readed from the texture.
	 *	@details	This struct provides an interface for 1D layered CUDA texture object.
	 */
	template<typename Type> struct Tex1DLayered : public Texture
	{
		//	The base type without const or volatile qualifiers.
		using value_type = std::remove_cv_t<Type>;

		//	Default constructor.
		CX_CUDA_CALLABLE Tex1DLayered() : Texture() {}

		//	Constructor with nullptr.
		CX_CUDA_CALLABLE Tex1DLayered(std::nullptr_t) : Texture(nullptr) {}

		//	Constructor with cudaTextureObject_t.
		CX_CUDA_CALLABLE explicit Tex1DLayered(cudaTextureObject_t hTexture) : Texture(hTexture) {}

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type fetch(float x, int layer) const;
	#else
		__device__ value_type fetch(float x, int layer) const
		{
			InternalValueType<Type> value;

			tex1DLayered<decltype(value)>(&value, m_hTexture, x, layer);

			return reinterpret_cast<value_type&>(value);
		}
	#endif
	};

	/*****************************************************************************
	************************    Tex1DLayeredLod<Type>    *************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D layered lod texture object for device access.
	 *	@tparam		Type - The data type readed from the texture.
	 *	@details	This struct provides an interface for 1D layered lod texture object.
	 */
	template<typename Type> struct Tex1DLayeredLod : public Texture
	{
		//	The base type without const or volatile qualifiers.
		using value_type = std::remove_cv_t<Type>;

		//	Default constructor.
		CX_CUDA_CALLABLE Tex1DLayeredLod() : Texture() {}

		//	Constructor with nullptr.
		CX_CUDA_CALLABLE Tex1DLayeredLod(std::nullptr_t) : Texture(nullptr) {}

		//	Constructor with cudaTextureObject_t.
		CX_CUDA_CALLABLE explicit Tex1DLayeredLod(cudaTextureObject_t hTexture) : Texture(hTexture) {}

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type fetch(float x, int layer, float level) const;
	#else
		__device__ value_type fetch(float x, int layer, float level) const
		{
			InternalValueType<Type> value;

			tex1DLayeredLod<decltype(value)>(&value, m_hTexture, x, layer, level);

			return reinterpret_cast<value_type&>(value);
		}
	#endif

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type grad(float x, int layer, float dx, float dy) const;
	#else
		__device__ value_type grad(float x, int layer, float dx, float dy) const
		{
			InternalValueType<Type> value;

			tex1DLayeredGrad<decltype(value)>(&value, m_hTexture, x, layer, dx, dy);

			return reinterpret_cast<value_type&>(value);
		}
	#endif
	};

	/*****************************************************************************
	*****************************    Tex2D<Type>    ******************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D texture object for device access.
	 *	@tparam		Type - The data type readed from the texture.
	 *	@details	This struct provides an interface for 2D texture object.
	 */
	template<typename Type> struct Tex2D : public Texture
	{
		//	The base type without const or volatile qualifiers.
		using value_type = std::remove_cv_t<Type>;

		//	Default constructor.
		CX_CUDA_CALLABLE Tex2D() : Texture() {}

		//	Constructor with nullptr.
		CX_CUDA_CALLABLE Tex2D(std::nullptr_t) : Texture(nullptr) {}

		//	Constructor with cudaTextureObject_t.
		CX_CUDA_CALLABLE explicit Tex2D(cudaTextureObject_t hTexture) : Texture(hTexture) {}

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type fetch(float x, float y) const;
	#else
		__device__ value_type fetch(float x, float y) const
		{
			InternalValueType<Type> value;

			tex2D<decltype(value)>(&value, m_hTexture, x, y);

			return reinterpret_cast<value_type&>(value);
		}
	#endif
	};

	/*****************************************************************************
	****************************    Tex2DLod<Type>    ****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D lod texture object for device access.
	 *	@tparam		Type - The data type readed from the texture.
	 *	@details	This struct provides an interface for 2D lod texture object.
	 */
	template<typename Type> struct Tex2DLod : public Texture
	{
		//	The base type without const or volatile qualifiers.
		using value_type = std::remove_cv_t<Type>;

		//	Default constructor.
		CX_CUDA_CALLABLE Tex2DLod() : Texture() {}

		//	Constructor with nullptr.
		CX_CUDA_CALLABLE Tex2DLod(std::nullptr_t) : Texture(nullptr) {}

		//	Constructor with cudaTextureObject_t.
		CX_CUDA_CALLABLE explicit Tex2DLod(cudaTextureObject_t hTexture) : Texture(hTexture) {}

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type fetch(float x, float y, float level) const;
	#else
		__device__ value_type fetch(float x, float y, float level) const
		{
			InternalValueType<Type> value;

			tex2DLod<decltype(value)>(&value, m_hTexture, x, y, level);

			return reinterpret_cast<value_type&>(value);
		}
	#endif

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type grad(float x, float y, CX_NAMESPACE::float2 dPdx, CX_NAMESPACE::float2 dPdy) const;
	#else
		__device__ value_type grad(float x, float y, CX_NAMESPACE::float2 dPdx, CX_NAMESPACE::float2 dPdy) const
		{
			InternalValueType<Type> value;

			tex2DGrad<decltype(value)>(&value, m_hTexture, x, y, reinterpret_cast<::float2&>(dPdx), reinterpret_cast<::float2&>(dPdy));

			return reinterpret_cast<value_type&>(value);
		}
	#endif
	};

	/*****************************************************************************
	**************************    Tex2DLayered<Type>    **************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D layered texture object for device access.
	 *	@tparam		Type - The data type readed from the texture.
	 *	@details	This struct provides an interface for 2D layered texture object.
	 */
	template<typename Type> struct Tex2DLayered : public Texture
	{
		//	The base type without const or volatile qualifiers.
		using value_type = std::remove_cv_t<Type>;

		//	Default constructor.
		CX_CUDA_CALLABLE Tex2DLayered() : Texture() {}

		//	Constructor with nullptr.
		CX_CUDA_CALLABLE Tex2DLayered(std::nullptr_t) : Texture(nullptr) {}

		//	Constructor with cudaTextureObject_t.
		CX_CUDA_CALLABLE explicit Tex2DLayered(cudaTextureObject_t hTexture) : Texture(hTexture) {}

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type fetch(float x, float y, int layer) const;
	#else
		__device__ value_type fetch(float x, float y, int layer) const
		{
			InternalValueType<Type> value;

			tex2DLayered<decltype(value)>(&value, m_hTexture, x, y, layer);

			return reinterpret_cast<value_type&>(value);
		}
	#endif
	};

	/*****************************************************************************
	************************    Tex2DLayeredLod<Type>    *************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D layered lod texture object for device access.
	 *	@tparam		Type - The data type readed from the texture.
	 *	@details	This struct provides an interface for 2D layered lod texture object.
	 */
	template<typename Type> struct Tex2DLayeredLod : public Texture
	{
		//	The base type without const or volatile qualifiers.
		using value_type = std::remove_cv_t<Type>;

		//	Default constructor.
		CX_CUDA_CALLABLE Tex2DLayeredLod() : Texture() {}

		//	Constructor with nullptr.
		CX_CUDA_CALLABLE Tex2DLayeredLod(std::nullptr_t) : Texture(nullptr) {}

		//	Constructor with cudaTextureObject_t.
		CX_CUDA_CALLABLE explicit Tex2DLayeredLod(cudaTextureObject_t hTexture) : Texture(hTexture) {}

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type fetch(float x, float y, int layer, float level) const;
	#else
		__device__ value_type fetch(float x, float y, int layer, float level) const
		{
			InternalValueType<Type> value;

			tex2DLayeredLod<decltype(value)>(&value, m_hTexture, x, y, layer, level);

			return reinterpret_cast<value_type&>(value);
		}
	#endif

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type grad(float x, float y, int layer, CX_NAMESPACE::float2 dPdx, CX_NAMESPACE::float2 dPdy) const;
	#else
		__device__ value_type grad(float x, float y, int layer, CX_NAMESPACE::float2 dPdx, CX_NAMESPACE::float2 dPdy) const
		{
			InternalValueType<Type> value;

			tex2DLayeredGrad<decltype(value)>(&value, m_hTexture, x, y, layer, reinterpret_cast<::float2&>(dPdx), reinterpret_cast<::float2&>(dPdy));

			return reinterpret_cast<value_type&>(value);
		}
	#endif
	};

	/*****************************************************************************
	*****************************    Tex3D<Type>    ******************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 3D texture object for device access.
	 *	@tparam		Type - The data type readed from the texture.
	 *	@details	This struct provides an interface for 2D texture object.
	 */
	template<typename Type> struct Tex3D : public Texture
	{
		//	The base type without const or volatile qualifiers.
		using value_type = std::remove_cv_t<Type>;

		//	Default constructor.
		CX_CUDA_CALLABLE Tex3D() : Texture() {}

		//	Constructor with nullptr.
		CX_CUDA_CALLABLE Tex3D(std::nullptr_t) : Texture(nullptr) {}

		//	Constructor with cudaTextureObject_t.
		CX_CUDA_CALLABLE explicit Tex3D(cudaTextureObject_t hTexture) : Texture(hTexture) {}

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type fetch(float x, float y, float z) const;
	#else
		__device__ value_type fetch(float x, float y, float z) const
		{
			InternalValueType<Type> value;

			tex3D<decltype(value)>(&value, m_hTexture, x, y, z);

			return reinterpret_cast<value_type&>(value);
		}
	#endif
	};

	/*****************************************************************************
	****************************    Tex3DLod<Type>    ****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 3D lod texture object for device access.
	 *	@tparam		Type - The data type readed from the texture.
	 *	@details	This struct provides an interface for 3D lod texture object.
	 */
	template<typename Type> struct Tex3DLod : public Texture
	{
		//	The base type without const or volatile qualifiers.
		using value_type = std::remove_cv_t<Type>;

		//	Default constructor.
		CX_CUDA_CALLABLE Tex3DLod() : Texture() {}

		//	Constructor with nullptr.
		CX_CUDA_CALLABLE Tex3DLod(std::nullptr_t) : Texture(nullptr) {}

		//	Constructor with cudaTextureObject_t.
		CX_CUDA_CALLABLE explicit Tex3DLod(cudaTextureObject_t hTexture) : Texture(hTexture) {}

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type fetch(float x, float y, float z, float level) const;
	#else
		__device__ value_type fetch(float x, float y, float z, float level) const
		{
			InternalValueType<Type> value;

			tex3DLod<decltype(value)>(&value, m_hTexture, x, y, z, level);

			return reinterpret_cast<value_type&>(value);
		}
	#endif

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type grad(float x, float y, float z, CX_NAMESPACE::float4 dPdx, CX_NAMESPACE::float4 dPdy) const;
	#else
		__device__ value_type grad(float x, float y, float z, CX_NAMESPACE::float4 dPdx, CX_NAMESPACE::float4 dPdy) const
		{
			InternalValueType<Type> value;

			tex3DGrad<decltype(value)>(&value, m_hTexture, x, y, z, reinterpret_cast<::float4&>(dPdx), reinterpret_cast<::float4&>(dPdy));

			return reinterpret_cast<value_type&>(value);
		}
	#endif
	};

	/*****************************************************************************
	****************************    TexCube<Type>    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a cube texture object for device access.
	 *	@tparam		Type - The data type readed from the texture.
	 *	@details	This struct provides an interface for cube texture object.
	 */
	template<typename Type> struct TexCube : public Texture
	{
		//	The base type without const or volatile qualifiers.
		using value_type = std::remove_cv_t<Type>;

		//	Default constructor.
		CX_CUDA_CALLABLE TexCube() : Texture() {}

		//	Constructor with nullptr.
		CX_CUDA_CALLABLE TexCube(std::nullptr_t) : Texture(nullptr) {}

		//	Constructor with cudaTextureObject_t.
		CX_CUDA_CALLABLE explicit TexCube(cudaTextureObject_t hTexture) : Texture(hTexture) {}

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type fetch(float x, float y, float z) const;
	#else
		__device__ value_type fetch(float x, float y, float z) const
		{
			InternalValueType<Type> value;

			texCubemap<decltype(value)>(&value, m_hTexture, x, y, z);

			return reinterpret_cast<value_type&>(value);
		}
	#endif
	};

	/*****************************************************************************
	***************************    TexCubeLod<Type>    ***************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a cube lod texture object for device access.
	 *	@tparam		Type - The data type readed from the texture.
	 *	@details	This struct provides an interface for cube lod texture object.
	 */
	template<typename Type> struct TexCubeLod : public Texture
	{
		//	The base type without const or volatile qualifiers.
		using value_type = std::remove_cv_t<Type>;

		//	Default constructor.
		CX_CUDA_CALLABLE TexCubeLod() : Texture() {}

		//	Constructor with nullptr.
		CX_CUDA_CALLABLE TexCubeLod(std::nullptr_t) : Texture(nullptr) {}

		//	Constructor with cudaTextureObject_t.
		CX_CUDA_CALLABLE explicit TexCubeLod(cudaTextureObject_t hTexture) : Texture(hTexture) {}

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type fetch(float x, float y, float z, float level) const;
	#else
		__device__ value_type fetch(float x, float y, float z, float level) const
		{
			InternalValueType<Type> value;

			texCubemapLod<decltype(value)>(&value, m_hTexture, x, y, z, level);

			return reinterpret_cast<value_type&>(value);
		}
	#endif

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type grad(float x, float y, float z, CX_NAMESPACE::float4 dPdx, CX_NAMESPACE::float4 dPdy) const;
	#else
		__device__ value_type grad(float x, float y, float z, CX_NAMESPACE::float4 dPdx, CX_NAMESPACE::float4 dPdy) const
		{
			InternalValueType<Type> value;

			texCubemapGrad<decltype(value)>(&value, m_hTexture, x, y, z, reinterpret_cast<::float4&>(dPdx), reinterpret_cast<::float4&>(dPdy));

			return reinterpret_cast<value_type&>(value);
		}
	#endif
	};

	/*****************************************************************************
	*************************    TexCubeLayered<Type>    *************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a cube layered texture object for device access.
	 *	@tparam		Type - The data type readed from the texture.
	 *	@details	This struct provides an interface for cube layered texture object.
	 */
	template<typename Type> struct TexCubeLayered : public Texture
	{
		//	The base type without const or volatile qualifiers.
		using value_type = std::remove_cv_t<Type>;

		//	Default constructor.
		CX_CUDA_CALLABLE TexCubeLayered() : Texture() {}

		//	Constructor with nullptr.
		CX_CUDA_CALLABLE TexCubeLayered(std::nullptr_t) : Texture(nullptr) {}

		//	Constructor with cudaTextureObject_t.
		CX_CUDA_CALLABLE explicit TexCubeLayered(cudaTextureObject_t hTexture) : Texture(hTexture) {}

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type fetch(float x, float y, float z, int layer) const;
	#else
		__device__ value_type fetch(float x, float y, float z, int layer) const
		{
			InternalValueType<Type> value;

			texCubemapLayered<decltype(value)>(&value, m_hTexture, x, y, z, layer);

			return reinterpret_cast<value_type&>(value);
		}
	#endif
	};

	/*****************************************************************************
	***********************    TexCubeLayeredLod<Type>    ************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a cube layered lod texture object for device access.
	 *	@tparam		Type - The data type readed from the texture.
	 *	@details	This struct provides an interface for cube layered lod texture object.
	 */
	template<typename Type> struct TexCubeLayeredLod : public Texture
	{
		//	The base type without const or volatile qualifiers.
		using value_type = std::remove_cv_t<Type>;

		//	Default constructor.
		CX_CUDA_CALLABLE TexCubeLayeredLod() : Texture() {}

		//	Constructor with nullptr.
		CX_CUDA_CALLABLE TexCubeLayeredLod(std::nullptr_t) : Texture(nullptr) {}

		//	Constructor with cudaTextureObject_t.
		CX_CUDA_CALLABLE explicit TexCubeLayeredLod(cudaTextureObject_t hTexture) : Texture(hTexture) {}

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ value_type fetch(float x, float y, float z, int layer, float level) const;
	#else
		__device__ value_type fetch(float x, float y, float z, int layer, float level) const
		{
			InternalValueType<Type> value;

			texCubemapLayeredLod<decltype(value)>(&value, m_hTexture, x, y, z, layer, level);

			return reinterpret_cast<value_type&>(value);
		}
	#endif

		//	Read method for CUDA texture object.
	#ifndef __CUDACC__
		__device__ Type grad(float x, float y, float z, int layer, CX_NAMESPACE::float4 dPdx, CX_NAMESPACE::float4 dPdy) const;
	#else
		__device__ Type grad(float x, float y, float z, int layer, CX_NAMESPACE::float4 dPdx, CX_NAMESPACE::float4 dPdy) const
		{
			InternalValueType<Type> value;

			texCubemapLayeredGrad<decltype(value)>(&value, m_hTexture, x, y, z, layer, reinterpret_cast<::float4&>(dPdx), reinterpret_cast<::float4&>(dPdy));

			return reinterpret_cast<value_type&>(value);
		}
	#endif
	};
}