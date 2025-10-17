#include "format.h"
#include "logger.h"
#include "device.h"
#include "allocator.h"
#include <cuda_runtime.h>

CX_USING_NAMESPACE

/*********************************************************************************
******************************    HostAllocator    *******************************
*********************************************************************************/

void * HostAllocator::doAllocateMemory(size_t bytes)
{
	void * ptr = nullptr;

	cudaError_t err = cudaMallocHost(&ptr, bytes);

	if (err != cudaSuccess)
	{
		CX_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}

#ifdef CX_CUDA_MEMORY_CHECK
	err = cudaMemset(ptr, -1, bytes);
#endif

	return ptr;
}


void HostAllocator::doDeallocateMemory(void * ptr)
{
	cudaError_t err = cudaFreeHost(ptr);

	if (err != cudaSuccess)
	{
		CX_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}

/*********************************************************************************
*****************************    DeviceAllocator    ******************************
*********************************************************************************/

DeviceAllocator::DeviceAllocator(class Device * pDevice) : m_device(pDevice)
{
	CX_ASSERT(pDevice != nullptr);
}


static cudaChannelFormatDesc createChannelDesc(Format format)
{
	cudaChannelFormatDesc channelDesc = {};

	switch (format)
	{
		case Format::Int:		{ channelDesc = cudaCreateChannelDesc<::int1>();		break; }
		case Format::Int2:		{ channelDesc = cudaCreateChannelDesc<::int2>();		break; }
		case Format::Int4:		{ channelDesc = cudaCreateChannelDesc<::int4>();		break; }
		case Format::Uint:		{ channelDesc = cudaCreateChannelDesc<::uint1>();		break; }
		case Format::Uint2:		{ channelDesc = cudaCreateChannelDesc<::uint2>();		break; }
		case Format::Uint4:		{ channelDesc = cudaCreateChannelDesc<::uint4>();		break; }
		case Format::Char:		{ channelDesc = cudaCreateChannelDesc<::char1>();		break; }
		case Format::Char2:		{ channelDesc = cudaCreateChannelDesc<::char2>();		break; }
		case Format::Char4:		{ channelDesc = cudaCreateChannelDesc<::char4>();		break; }
		case Format::Uchar:		{ channelDesc = cudaCreateChannelDesc<::uchar1>();		break; }
		case Format::Uchar2:	{ channelDesc = cudaCreateChannelDesc<::uchar2>();		break; }
		case Format::Uchar4:	{ channelDesc = cudaCreateChannelDesc<::uchar4>();		break; }
		case Format::Short:		{ channelDesc = cudaCreateChannelDesc<::short1>();		break; }
		case Format::Short2:	{ channelDesc = cudaCreateChannelDesc<::short2>();		break; }
		case Format::Short4:	{ channelDesc = cudaCreateChannelDesc<::short4>();		break; }
		case Format::Ushort:	{ channelDesc = cudaCreateChannelDesc<::ushort1>();		break; }
		case Format::Ushort2:	{ channelDesc = cudaCreateChannelDesc<::ushort2>();		break; }
		case Format::Ushort4:	{ channelDesc = cudaCreateChannelDesc<::ushort4>();		break; }
		case Format::Float:		{ channelDesc = cudaCreateChannelDesc<::float1>();		break; }
		case Format::Float2:	{ channelDesc = cudaCreateChannelDesc<::float2>();		break; }
		case Format::Float4:	{ channelDesc = cudaCreateChannelDesc<::float4>();		break; }
		default:				{ CX_ERROR_LOG("Invalid format!");						break; }
	}

	return channelDesc;
}


cudaArray_t DeviceAllocator::allocateTextureMemory(Format format, size_t width, size_t height, size_t depth, int flags)
{
	m_device->setCurrent();

	cudaArray_t hArray = nullptr;

	cudaChannelFormatDesc channelDesc = createChannelDesc(format);

	cudaError_t err = cudaMalloc3DArray(&hArray, &channelDesc, make_cudaExtent(width, height, depth), flags);

	if (err != cudaSuccess)
	{
		CX_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}

	return hArray;
}


cudaMipmappedArray_t DeviceAllocator::allocateMipmapTextureMemory(Format format, size_t width, size_t height, size_t depth, unsigned int numLevels, int flags)
{
	m_device->setCurrent();

	cudaMipmappedArray_t hMipmapedArray = nullptr;

	cudaChannelFormatDesc channelDesc = createChannelDesc(format);

	cudaError_t err = cudaMallocMipmappedArray(&hMipmapedArray, &channelDesc, make_cudaExtent(width, height, depth), numLevels, flags);

	if (err != cudaSuccess)
	{
		CX_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}

	return hMipmapedArray;
}


void DeviceAllocator::deallocateMipmapTextureMemory(cudaMipmappedArray_t hMipmapedArray)
{
	cudaError_t err = cudaFreeMipmappedArray(hMipmapedArray);

	if (err != cudaSuccess)
	{
		CX_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}


void DeviceAllocator::deallocateTextureMemory(cudaArray_t hArray)
{
	cudaError_t err = cudaFreeArray(hArray);

	if (err != cudaSuccess)
	{
		CX_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}


void * DeviceAllocator::doAllocateMemory(size_t bytes)
{
	void * ptr = nullptr;

	m_device->setCurrent();

	cudaError_t err = cudaMalloc(&ptr, bytes);

	if (err != cudaSuccess)
	{
		CX_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}

#ifdef CX_CUDA_MEMORY_CHECK
	err = cudaMemset(ptr, -1, bytes);

	cudaDeviceSynchronize();
#endif

	return ptr;
}


void DeviceAllocator::doDeallocateMemory(void * ptr)
{
	cudaError_t err = cudaFree(ptr);

	if (err != cudaSuccess)
	{
		CX_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}