
#pragma once

#include "fwd.h"
#include "host_types.h"
#include <vector>

namespace CX_NAMESPACE
{
	/*****************************************************************************
	*******************************    Context    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Wrapper for CUDA context object (singleton).
	 */
	class Context
	{
		CX_NONCOPYABLE(Context)

	private:

		//!	@brief		Create CUDA context wrapper.
		Context();

		//!	@brief		Destroy CUDA context wrapper.
		~Context();

	public:

		/**
		 *	@brief		Return a raw pointer to the CUDA context wrapper (singleton).
		 */
		static Context * getInstance()
		{
			static Context s_instance;

			return &s_instance;
		}

	public:

		/**
		 *	@brief		Return the last error from a runtime call.
		 *	@note		Return the last error that has been produced by any of the runtime calls
		 *				in the same host thread and reset it to Error::eSuccess.
		 */
		static cudaError_t getLastError() noexcept;


		/**
		 *	@brief		Return a string containing the name of an error code in the enum.
		 *	@note		If the error code is not recognized, "unrecognized error code" is returned.
		 */
		static const char * getErrorName(cudaError_t eValue) noexcept;


		/**
		 *	@brief		Return the description string for an error code.
		 *	@note		If the error code is not recognized, "unrecognized error code" is returned.
		 */
		static const char * getErrorString(cudaError_t eValue) noexcept;

	public:

		/**
		 *	@brief		Return the latest version of CUDA supported by the driver.
		 */
		Version driverVersion() const { return m_driverVersion; }


		/**
		 *	@brief		Return the version number of the current CUDA Runtime instance.
		 */
		Version runtimeVersion() const { return m_runtimeVersion; }


		/**
		 *	@brief		Return pointer to physical device.
		 */
		Device * device(size_t index) const { return m_pNvidiaDevices[index]; }


		/**
		 *	@brief		Return physical device array.
		 */
		const std::vector<Device*> & getDevices() const { return m_pNvidiaDevices; }

	private:

		Version						m_driverVersion;
		Version						m_runtimeVersion;
		std::vector<Device*>		m_pNvidiaDevices;
	};
}