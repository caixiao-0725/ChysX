

#include "logger.h"
#include "device.h"
#include "context.h"
#include <cuda_runtime_api.h>

CX_USING_NAMESPACE

/*********************************************************************************
*********************************    Context    **********************************
*********************************************************************************/

Context::Context()
{
	cudaGetLastError();

	//////////////////////////////////////////////////////////////////////

	int driverVersion = 0;

	cudaDriverGetVersion(&driverVersion);

	m_driverVersion.Major = driverVersion / 1000;
	m_driverVersion.Minor = (driverVersion % 1000) / 10;

	CX_INFO_LOG("CUDA driver version: %d.%d", m_driverVersion.Major, m_driverVersion.Minor);

	//////////////////////////////////////////////////////////////////////

	int runtimeVersion = 0;

	cudaRuntimeGetVersion(&runtimeVersion);

	m_runtimeVersion.Major = runtimeVersion / 1000;
	m_runtimeVersion.Minor = (runtimeVersion % 1000) / 10;

	CX_INFO_LOG("CUDA runtime version: %d.%d", m_runtimeVersion.Major, m_runtimeVersion.Minor);

	//////////////////////////////////////////////////////////////////////

	cudaGetLastError();

	int deviceCount = 0;

	auto err = cudaGetDeviceCount(&deviceCount);

	m_pNvidiaDevices.resize(deviceCount, nullptr);

	CX_INFO_LOG_IF(err == cudaErrorNoDevice, "No CUDA-capable devices were detected.");

	//////////////////////////////////////////////////////////////////////

	for (int i = 0; i < deviceCount; i++)
	{
		cudaDeviceProp devProp = {};

		cudaGetDeviceProperties(reinterpret_cast<cudaDeviceProp*>(&devProp), i);

		CX_INFO_LOG("CUDA device(%d): %s, compute capability: %d.%d", i, devProp.name, devProp.major, devProp.minor);

		m_pNvidiaDevices[i] = new Device(i, devProp);
	}

	cudaGetLastError();
}


const char * Context::getErrorString(cudaError_t eValue) noexcept
{
	return cudaGetErrorString(eValue);
}


const char * Context::getErrorName(cudaError_t eValue) noexcept
{
	return cudaGetErrorName(eValue);
}


cudaError_t Context::getLastError() noexcept
{
	return cudaGetLastError();
}


Context::~Context()
{
	for (size_t i = 0; i < m_pNvidiaDevices.size(); i++)
	{
		delete m_pNvidiaDevices[i];
	}

	m_pNvidiaDevices.clear();
}