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

#include <cudaAPI/device.h>
#include <cudaAPI/context.h>
#include <cudaAPI/buffer_view.h>

 /*********************************************************************************
 *****************************    buffer_view_test    *****************************
 *********************************************************************************/

void buffer_view_test()
{
	auto device = cx::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();
	auto buffer = std::make_shared<cx::Buffer>(allocator, sizeof(int) * 1024);

	cx::BufferView1D<int> bufferView0;
	cx::BufferView1D<int> bufferView1 = nullptr;
	cx::BufferView1D<int> bufferView2(buffer);
	cx::BufferView1D<int> bufferView3(buffer, 0, sizeof(int) * 10);

	cx::BufferView2D<int> bufferView4;
	cx::BufferView2D<int> bufferView5 = nullptr;

	cx::BufferView3D<int> bufferView6;
	cx::BufferView3D<int> bufferView7 = nullptr;

	cx::view_cast<int>(bufferView0);
	cx::view_cast<float>(bufferView4);
	cx::view_cast<unsigned int>(bufferView6);
}