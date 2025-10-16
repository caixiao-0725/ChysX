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
#include "event.h"
#include "stream.h"
#include <functional>

namespace CX_NAMESPACE
{
	/*****************************************************************************
	*****************************    ScopedTimer    ******************************
	*****************************************************************************/

	/**
	 *	@brief		RAII-style GPU timer that measures execution duration between stream events.
	 *	@details	Automatically records start/end events on construction/destruction and invokes
	 *				a callback with the measured duration. Designed for CUDA stream timing.
	 *	@note		Designed specifically for development/debugging purposes to profile CUDA stream operations.
	 *				Not recommended for production code due to synchronization overhead.
	 */
	class ScopedTimer
	{
		CX_NONCOPYABLE(ScopedTimer)

	public:

		//	Callback type that receives the measured duration in microseconds.
		using CallbackFunc = std::function<void(float us)>;


		/**
		 *	@brief		Constructs timer and records start event.
		 *	@param[in]	stream - The CUDA stream to time operations on.
		 *	@param[in]	callback - Function to receive timing results.
		 */
		explicit ScopedTimer(Stream & stream, const CallbackFunc & callback) : m_stream(stream),
			m_callback(callback), m_startEvent(stream.device()), m_endEvent(stream.device())
		{
			m_stream.recordEvent(m_startEvent);
		}


		/**
		 *	@brief		Constructs timer and records start event.
		 *	@param[in]	stream - The CUDA stream to time operations on.
		 *	@param[in]	callback - Function to receive timing results.
		 */
		explicit ScopedTimer(Stream * stream, const CallbackFunc & callback) : ScopedTimer(*stream, callback) {}


		/**
		 *	@brief		Constructs timer and records start event.
		 *	@param[in]	stream - The CUDA stream to time operations on.
		 *	@param[in]	callback - Function to receive timing results.
		 */
		explicit ScopedTimer(std::shared_ptr<Stream> stream, const CallbackFunc & callback) : ScopedTimer(*stream, callback) {}


		/**
		 *	@brief		Destructor - records end event and invokes callback
		 *	@details	Synchronizes the stream before measurement to ensure accurate timing.
		 *				The callback receives the elapsed time between start/end events.
		 */
		~ScopedTimer()
		{
			m_stream.recordEvent(m_endEvent).sync();

			auto time = TimedEvent::elapsedTime(m_startEvent, m_endEvent);

			m_callback(static_cast<float>(time.count() * 1e-3f));
		}

	private:

		Stream &			m_stream;
		TimedEvent			m_endEvent;
		TimedEvent			m_startEvent;
		CallbackFunc		m_callback;
	};
}