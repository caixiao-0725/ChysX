#include "image.h"
#include "logger.h"
#include "texture.h"
#include "image_cube.h"
#include <cuda_runtime_api.h>

CX_USING_NAMESPACE

/*********************************************************************************
*******************************    Validations    ********************************
*********************************************************************************/

static_assert(static_cast<int>(FilterMode::Linear) == cudaFilterModeLinear);
static_assert(static_cast<int>(FilterMode::Nearest) == cudaFilterModePoint);

static_assert(static_cast<int>(AddressMode::Wrap) == cudaAddressModeWrap);
static_assert(static_cast<int>(AddressMode::Clamp) == cudaAddressModeClamp);
static_assert(static_cast<int>(AddressMode::Mirror) == cudaAddressModeMirror);
static_assert(static_cast<int>(AddressMode::Border) == cudaAddressModeBorder);

/*********************************************************************************
*********************************    Texture    **********************************
*********************************************************************************/

Texture::Texture() : m_hTexture(0)
{

}


void Texture::bindImage(std::shared_ptr<ImageBase> image, Sampler sampler, Format viewFormat)
{
	this->unbind();

	auto pImage = std::dynamic_pointer_cast<Image>(image);
	auto pImageLod = std::dynamic_pointer_cast<ImageLod>(image);

	CX_ASSERT(pImage || pImageLod);
	{
		cudaResourceDesc resDesc = {};

		Format format = Format::Char;

		if (pImage != nullptr)
		{
			format = pImage->format();

			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = pImage->data().handle;

			if (!std::dynamic_pointer_cast<ImageCube<void>>(pImage) &&
				!std::dynamic_pointer_cast<ImageCubeLayered<void>>(pImage))
			{
				sampler.seamlessCubemap = false;	//	avoid error
			}
		}
		else if (pImageLod != nullptr)
		{
			format = pImageLod->format();

			resDesc.resType = cudaResourceTypeMipmappedArray;
			resDesc.res.mipmap.mipmap = pImageLod->handle();

			if (!std::dynamic_pointer_cast<ImageCubeLod<void>>(pImageLod) &&
				!std::dynamic_pointer_cast<ImageCubeLayeredLod<void>>(pImageLod))
			{
				sampler.seamlessCubemap = false;	//	avoid error
			}
		}

		if ((viewFormat != Format::Float) && (viewFormat != Format::Float2) && (viewFormat != Format::Float4))
		{
			sampler.mipmapFilterMode = FilterMode::Nearest;

			sampler.filterMode = FilterMode::Nearest;
		}

		cudaTextureDesc								textureDesc = {};
		textureDesc.addressMode[0]					= static_cast<cudaTextureAddressMode>(sampler.addressMode[0]);
		textureDesc.addressMode[1]					= static_cast<cudaTextureAddressMode>(sampler.addressMode[1]);
		textureDesc.addressMode[2]					= static_cast<cudaTextureAddressMode>(sampler.addressMode[2]);
		textureDesc.filterMode						= static_cast<cudaTextureFilterMode>(sampler.filterMode);
		textureDesc.readMode						= (format == viewFormat) ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
		textureDesc.sRGB							= sampler.sRGB;
		textureDesc.borderColor[0]					= sampler.borderColor[0];
		textureDesc.borderColor[1]					= sampler.borderColor[1];
		textureDesc.borderColor[2]					= sampler.borderColor[2];
		textureDesc.borderColor[3]					= sampler.borderColor[3];
		textureDesc.normalizedCoords				= sampler.normalizedCoords;
		textureDesc.maxAnisotropy					= sampler.maxAnisotropy;
		textureDesc.mipmapFilterMode				= static_cast<cudaTextureFilterMode>(sampler.mipmapFilterMode);
		textureDesc.mipmapLevelBias					= sampler.mipmapLevelBias;
		textureDesc.minMipmapLevelClamp				= sampler.minMipmapLevelClamp;
		textureDesc.maxMipmapLevelClamp				= sampler.maxMipmapLevelClamp;
		textureDesc.disableTrilinearOptimization	= sampler.disableTrilinearOptimization;
		textureDesc.seamlessCubemap					= sampler.seamlessCubemap;

		cudaError_t err = cudaCreateTextureObject(&m_hTexture, &resDesc, &textureDesc, nullptr);

		if (err == cudaSuccess)
		{
			m_sampler = sampler;

			m_image = image;
		}
		else
		{
			CX_ERROR_LOG("%s.", cudaGetErrorString(err));

			cudaGetLastError();

			throw err;
		}
	}
}


void Texture::unbind() noexcept
{
	if (m_hTexture != 0)
	{
		cudaError_t err = cudaDestroyTextureObject(m_hTexture);

		if (err != cudaSuccess)
		{
			CX_ERROR_LOG("%s.", cudaGetErrorString(err));

			cudaGetLastError();
		}

		m_image = nullptr;

		m_hTexture = 0;
	}
}


Texture::~Texture()
{
	this->unbind();
}