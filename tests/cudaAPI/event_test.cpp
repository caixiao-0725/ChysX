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

#include <cudaAPI/event.h>
#include <cudaAPI/device.h>
#include <cudaAPI/stream.h>
#include <cudaAPI/context.h>
#include <cudaAPI/scoped_timer.h>

/*********************************************************************************
********************************    event_test    ********************************
*********************************************************************************/

void event_test()
{
	auto device = cx::Context::getInstance()->device(0);
	auto & stream = device->defaultStream();

	cx::Event		event0(device);
	cx::TimedEvent	event1(device);
	cx::TimedEvent	event2(device);
	cx::ScopedTimer	timer(stream, [](float us) { printf("ScopedTime: %fus\n", us); });

	stream.recordEvent(event0);
	stream.recordEvent(event1);
	stream.recordEvent(event2);
	stream.waitEvent(event1).sync();

	auto time = cx::TimedEvent::elapsedTime(event1, event2);

	if (event1.query())
	{
		event1.sync();
		event1.device();
		event1.handle();
	}
}