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

#include <cudaAPI/logger.h>

/*********************************************************************************
*******************************    logger_test    ********************************
*********************************************************************************/

void logger_test()
{
	CX_INFO_LOG("This is info log.");
	CX_DEBUG_LOG("This is debug log.");
	CX_ERROR_LOG("This is error log.");
	CX_ASSERT_LOG("This is assert log");
	CX_WARNING_LOG("This is warning log.");

	cx::Logger::getInstance()->registerCallback([](const char * fileName, int line, const char * funcName, cx::Logger::Level level, const char * logMsg)
	{
		int a = 0;
	});

	CX_INFO_LOG_IF(true, "This is info log.");
	CX_DEBUG_LOG_IF(true, "This is debug log.");
	CX_ERROR_LOG_IF(true, "This is error log.");
//	CX_ASSERT_LOG_IF(true, "This is assert log");
	CX_WARNING_LOG_IF(true, "This is warning log.");
}