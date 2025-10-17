

#include "logger.h"
#include "device.h"
#include "stream.h"
#include "allocator.h"
#include <cuda_runtime_api.h>

CX_USING_NAMESPACE

/*********************************************************************************
**********************************    Device    **********************************
*********************************************************************************/

Device::Device(int deviceID, const cudaDeviceProp & devProp) : m_deviceID(deviceID), m_devProp(std::make_unique<cudaDeviceProp>(devProp)),
	m_defaultAlloc(std::make_shared<DeviceAllocator>(this)), m_defaultStream(new Stream(this, nullptr))
{

}


cudaError_t Device::init() noexcept
{
	this->setCurrent();

	cudaError_t err = cudaFree(nullptr);

	if (err != cudaSuccess)
	{
		CX_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return err;
}


size_t Device::freeMemorySize() const
{
	this->setCurrent();

	size_t freeMemInBytes = 0, totalMemInBytes = 0;

	cudaError_t err = cudaMemGetInfo(&freeMemInBytes, &totalMemInBytes);

	if (err != cudaSuccess)
	{
		CX_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return freeMemInBytes;
}


void Device::sync() const
{
	this->setCurrent();

	cudaError_t err = cudaDeviceSynchronize();

	if (err != cudaSuccess)
	{
		CX_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}


void Device::setCurrent() const
{
	thread_local int currentDevice = 0;

	if (this->m_deviceID != currentDevice)
	{
		cudaError_t err = cudaSetDevice(this->m_deviceID);

		if (err != cudaSuccess)
		{
			CX_ERROR_LOG("%s.", cudaGetErrorString(err));

			cudaGetLastError();
		}
		else
		{
			currentDevice = this->m_deviceID;
		}
	}
}


Device::~Device() noexcept
{

}