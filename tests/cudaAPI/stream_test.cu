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

#include <vector>
#include <functional>
#include <cudaAPI/stream.h>
#include <cudaAPI/device.h>
#include <cudaAPI/context.h>
#include <cudaAPI/array_1d.h>
#include <cudaAPI/array_2d.h>
#include <cudaAPI/array_3d.h>
#include <cudaAPI/image_1d.h>
#include <cudaAPI/image_2d.h>
#include <cudaAPI/image_3d.h>
#include <cudaAPI/launch_utils.cuh>
#include <device_launch_parameters.h>

/*********************************************************************************
*******************************    stream_event    *******************************
*********************************************************************************/

__constant__ int cache[100];

__global__ void test_kernel()
{
	CUDA_for(i, 1);

	printf("device: Happy Nucleus!\n");
}


void stream_test()
{
	auto device = cx::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();

	stream.sync();
	stream.query();
	stream.forceSync(true);
	stream.forceSync(false);
	assert(stream.device() == device);
	assert(stream.handle() == nullptr);

	int a;
	auto pfnTask = [](int*) { printf("host: Happy Nucleus!\n"); };
	stream.launchHostFunc<int>(pfnTask, &a);
	stream.launch(test_kernel, cx::ceil_div(15, 32), 32)();

	std::vector<int>	host_data(100, 33);
	cx::Array<int>		dev_data1(allocator, 100);
	cx::Array2D<int>	dev_data2(allocator, 10, 10);
	cx::Array3D<int>	dev_data3(allocator, 2, 5, 10);
	cx::Image1D<int>	dev_data4(allocator, 100);
	cx::Image2D<int>	dev_data5(allocator, 10, 10);
	cx::Image3D<int>	dev_data6(allocator, 2, 5, 10);

	stream.memset(dev_data1.data(), 1, dev_data1.size());
	stream.memcpy(host_data.data(), dev_data1.data(), dev_data1.size());

	for (size_t i = 0; i < host_data.size(); i++)
	{
		assert(host_data[i] == 1);
	}

	stream.memset(dev_data2.data(), 2, dev_data2.size());
	stream.memcpy2D(host_data.data(), dev_data2.pitch(), dev_data2.data(), dev_data2.pitch(), dev_data2.width(), dev_data2.height());
	
	for (size_t i = 0; i < host_data.size(); i++)
	{
		assert(host_data[i] == 2);
	}

	stream.memset(dev_data3.data(), 3, dev_data3.size());
	stream.memcpy3D(host_data.data(), dev_data3.pitch(), dev_data3.height(), dev_data3.data(), dev_data3.pitch(), dev_data3.height(), dev_data3.width(), dev_data3.height(), dev_data3.depth());

	for (size_t i = 0; i < host_data.size(); i++)
	{
		assert(host_data[i] == 3);
	}

	stream.memset(dev_data1.data(), 4, dev_data1.size());
	stream.memcpy(dev_data4.data(), dev_data1.data(), dev_data1.size());
	stream.memcpy(host_data.data(), dev_data4.data(), dev_data1.size());

	for (size_t i = 0; i < host_data.size(); i++)
	{
		assert(host_data[i] == 4);
	}

	stream.memset(dev_data2.data(), 5, dev_data2.size());
	stream.memcpy2D(dev_data5.data(), dev_data2.data(), dev_data2.pitch(), dev_data2.width(), dev_data2.height());
	stream.memcpy2D(host_data.data(), dev_data2.pitch(), dev_data5.data(), dev_data2.width(), dev_data2.height());

	for (size_t i = 0; i < host_data.size(); i++)
	{
		assert(host_data[i] == 5);
	}

	stream.memset(dev_data3.data(), 6, dev_data3.size());
	stream.memcpy3D(dev_data6.data(), dev_data3.data(), dev_data3.pitch(), dev_data3.height(), dev_data3.width(), dev_data3.height(), dev_data3.depth());
	stream.memcpy3D(host_data.data(), dev_data3.pitch(), dev_data3.height(), dev_data6.data(), dev_data3.width(), dev_data3.height(), dev_data3.depth());

	for (size_t i = 0; i < host_data.size(); i++)
	{
		assert(host_data[i] == 6);
	}

	host_data.assign(100, 7);
	stream.memcpyToSymbol(cache, host_data.data(), host_data.size());
	host_data.assign(100, 0);
	stream.memcpyFromSymbol(host_data.data(), cache, host_data.size());

	for (size_t i = 0; i < host_data.size(); i++)
	{
		assert(host_data[i] == 7);
	}
}