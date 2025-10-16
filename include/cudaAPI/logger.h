
#pragma once

#include "macros.h"
#include <functional>

namespace CX_NAMESPACE
{
	/*****************************************************************************
	********************************    Logger    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Lightweight logger (singleton).
	 */
	class Logger
	{
		CX_NONCOPYABLE(Logger)

	private:

		//!	@brief		Default constructor.
		Logger() = default;

		//!	@brief		Default destructor.
		~Logger() = default;

	public:

		/**
		 *	@brief		Log levels.
		 */
		enum Level { Assert, Error, Warning, Info, Debug };


		/**
		 *	@brief		Type of message callback function.
		 */
		using LogCallback = std::function<void(const char * fileName, int line, const char * funcName, Level level, const char * logMsg)>;

	public:

		/**
		 *	@brief		Return the singleton instance.
		 *	@warning	Be cautious when multiple dynamic libraries call this function (no longer a singleton in that case).
		 */
		static Logger * getInstance()
		{
			static Logger s_instance;

			return &s_instance;
		}


		/**
		 *	@brief		Specify a callback function.
		 *	@param[in]	pfnCallback - User-defined callback function for receiving log message.
		 *	@note		Pass nullptr will unregister the callback function.
		 */
		void registerCallback(LogCallback pfnCallback) noexcept { m_pfnCallback = pfnCallback; }


		/**
		 *	@brief		Transfer log message to the specified callback function (if registered).
		 *	@param[in]	fileName - Which file sends this message.
		 *	@param[in]	line - Which line in the file invokes this function.
		 *	@param[in]	funcName - Which function invokes this functin.
		 *	@param[in]	level - Log level of this message.
		 */
		void log(const char * fileName, int line, const char * funcName, Level level, const char * format, ...);

	private:

		LogCallback		m_pfnCallback;
	};
}

/*********************************************************************************
********************************    Log Macros    ********************************
*********************************************************************************/

#define CX_INFO_LOG(...)		CX_NAMESPACE::Logger::getInstance()->log(__FILE__, __LINE__, __FUNCTION__, CX_NAMESPACE::Logger::Info, __VA_ARGS__)
#define CX_DEBUG_LOG(...)		CX_NAMESPACE::Logger::getInstance()->log(__FILE__, __LINE__, __FUNCTION__, CX_NAMESPACE::Logger::Debug, __VA_ARGS__)
#define CX_ERROR_LOG(...)		CX_NAMESPACE::Logger::getInstance()->log(__FILE__, __LINE__, __FUNCTION__, CX_NAMESPACE::Logger::Error, __VA_ARGS__)
#define CX_ASSERT_LOG(...)		CX_NAMESPACE::Logger::getInstance()->log(__FILE__, __LINE__, __FUNCTION__, CX_NAMESPACE::Logger::Assert, __VA_ARGS__)
#define CX_WARNING_LOG(...)		CX_NAMESPACE::Logger::getInstance()->log(__FILE__, __LINE__, __FUNCTION__, CX_NAMESPACE::Logger::Warning, __VA_ARGS__)

#define CX_INFO_LOG_IF(condition, ...)			if (condition)	CX_INFO_LOG(__VA_ARGS__)
#define CX_ERROR_LOG_IF(condition, ...)			if (condition)	CX_ERROR_LOG(__VA_ARGS__)
#define CX_ASSERT_LOG_IF(condition, ...)		if (condition)	CX_ASSERT_LOG(__VA_ARGS__);		CX_ASSERT(!(condition))
#define CX_WARNING_LOG_IF(condition, ...)		if (condition)	CX_WARNING_LOG(__VA_ARGS__)

#ifdef CX_DEBUG
	#define CX_DEBUG_LOG_IF(condition, ...)		if (condition)	CX_DEBUG_LOG(__VA_ARGS__)
#else
	#define CX_DEBUG_LOG_IF(condition, ...)
#endif