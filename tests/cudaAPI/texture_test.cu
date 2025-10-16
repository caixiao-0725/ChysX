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

#include <cudaAPI/stream.h>
#include <cudaAPI/device.h>
#include <cudaAPI/context.h>
#include <cudaAPI/texture.h>
#include <cudaAPI/image_1d.h>
#include <cudaAPI/image_2d.h>
#include <cudaAPI/image_3d.h>
#include <cudaAPI/image_cube.h>
#include <cudaAPI/launch_utils.cuh>

/*********************************************************************************
*******************************    texture_test    *******************************
*********************************************************************************/

__global__ void test_device_texture(dev::Tex1D<int> tex0, dev::Tex1DLod<int> tex1, dev::Tex1DLayered<int> tex2, dev::Tex1DLayeredLod<int> tex3,
									dev::Tex2D<float> tex4, dev::Tex2DLod<int> tex5, dev::Tex2DLayered<int> tex6, dev::Tex2DLayeredLod<int> tex7,
									dev::Tex3D<int> tex8, dev::Tex3DLod<int> tex9,
									dev::TexCube<int> tex10, dev::TexCubeLod<int> tex11, dev::TexCubeLayered<int> tex12, dev::TexCubeLayeredLod<int> tex13)
{
	//	1D
	tex0.empty();
	tex0.fetch(0.5f);
	tex0.handle();

	tex1.empty();
	tex1.fetch(0.5f, 0);
	tex1.grad(0.0f, 1.0f, 2.0f);
	tex1.handle();

	tex2.empty();
	tex2.fetch(0.5f, 0);
	tex2.handle();
	
	tex3.empty();
	tex3.fetch(0.5f, 0, 0.5f);
	tex3.grad(0.0f, 1, 1.0f, 2.0f);
	tex3.handle();

	//	2D
	tex4.empty();
	tex4.fetch(0.5f, 0.5f);
	tex4.handle();

	tex5.empty();
	tex5.fetch(0.5f, 0.2f, 0);
	tex5.grad(0.0f, 1.0f, { 2.0f, 1.0f }, { 1.0f, 2.0f });
	tex5.handle();

	tex6.empty();
	tex6.fetch(0.5f, 0.0f, 0);
	tex6.handle();

	tex7.empty();
	tex7.fetch(0.5f, 1.0f, 0, 0.5f);
	tex7.grad(0.0f, 2.0f, 1, { 1.0f, 1.0f }, { 2.0f, 2.0f });
	tex7.handle();

	//	3D
	tex8.empty();
	tex8.fetch(0.5f, 0.5f, 2.0f);
	tex8.handle();

	tex9.empty();
	tex9.fetch(0.5f, 0.2f, 1.0f, 0);
	tex9.grad(0.0f, 1.0f, 2.0f, { 2.0f, 1.0f, 2.0f, 2.0f }, { 1.0f, 2.0f, 2.0f, 3.0f });
	tex9.handle();

	//	Cube
	tex10.empty();
	tex10.fetch(0.5f, 0.2f, 0.3f);
	tex10.handle();

	tex11.empty();
	tex11.fetch(0.5f, 0.6f, 0.8f, 0);
	tex11.grad(0.0f, 1.0f, 2.0f, { 2.0f, 1.0f, 2.0f, 2.0f }, { 1.0f, 2.0f, 2.0f, 3.0f });
	tex11.handle();

	tex12.empty();
	tex12.fetch(0.5f, 2.0f, 3.0f, 0);
	tex12.handle();

	tex13.empty();
	tex13.fetch(0.5f, 0.3f, 0.1f, 0, 0.5f);
	tex13.grad(0.5f, 0.3f, 0.1f, 1, { 2.0f, 1.0f, 2.0f, 2.0f }, { 1.0f, 2.0f, 2.0f, 3.0f });
	tex13.handle();
}


void texture_test()
{
	auto device = cx::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();

	//	1D
	cx::Texture1D<int>	texture0;
	texture0.bind(std::make_shared<cx::Image1D<int>>(allocator, 10));
	assert(texture0.image() != nullptr);
	
	cx::Texture1DLod<int>	texture1;
	texture1.bind(std::make_shared<cx::Image1DLod<int>>(allocator, 10, 2));
	assert(texture1.image() != nullptr);

	cx::Texture1DLayered<int>	texture2;
	texture2.bind(std::make_shared<cx::Image1DLayered<int>>(allocator, 10, 2));
	assert(texture2.image() != nullptr);

	cx::Texture1DLayeredLod<int>	texture3;
	texture3.bind(std::make_shared<cx::Image1DLayeredLod<int>>(allocator, 10, 2, 2));
	assert(texture3.image() != nullptr);

	//	2D
	cx::Texture2D<float>	texture4;
	texture4.bind(std::make_shared<cx::Image2D<char>>(allocator, 10, 10));
	texture4.bind(std::make_shared<cx::Image2D<float>>(allocator, 10, 10));
	assert(texture4.image() != nullptr);

	cx::Texture2DLod<int>	texture5;
	texture5.bind(std::make_shared<cx::Image2DLod<int>>(allocator, 10, 10, 2));
	assert(texture5.image() != nullptr);

	cx::Texture2DLayered<int>	texture6;
	texture6.bind(std::make_shared<cx::Image2DLayered<int>>(allocator, 10, 10, 2));
	assert(texture6.image() != nullptr);

	cx::Texture2DLayeredLod<int>	texture7;
	texture7.bind(std::make_shared<cx::Image2DLayeredLod<int>>(allocator, 10, 10, 2, 2));
	assert(texture7.image() != nullptr);

	//	3D
	cx::Texture3D<int>	texture8;
	texture8.bind(std::make_shared<cx::Image3D<int>>(allocator, 10, 10, 10));
	assert(texture8.image() != nullptr);

	cx::Texture3DLod<int>	texture9;
	texture9.bind(std::make_shared<cx::Image3DLod<int>>(allocator, 10, 10, 10, 2));
	assert(texture9.image() != nullptr);

	//	Cube
	cx::TextureCube<int>	texture10;
	texture10.bind(std::make_shared<cx::ImageCube<int>>(allocator, 10));
	assert(texture10.image() != nullptr);

	cx::TextureCubeLod<int>	texture11;
	texture11.bind(std::make_shared<cx::ImageCubeLod<int>>(allocator, 10, 2));
	assert(texture11.image() != nullptr);

	cx::TextureCubeLayered<int>	texture12;
	texture12.bind(std::make_shared<cx::ImageCubeLayered<int>>(allocator, 10, 2));
	assert(texture12.image() != nullptr);

	cx::TextureCubeLayeredLod<int>	texture13;
	texture13.bind(std::make_shared<cx::ImageCubeLayeredLod<int>>(allocator, 10, 2, 2));
	assert(texture13.image() != nullptr);

	// Read mode: normalized float
	cx::Texture2D<float>	texture14;
	texture14.bind(std::make_shared<cx::Image2D<char>>(allocator, 10, 10));
	texture14.bind(std::make_shared<cx::Image2D<short>>(allocator, 10, 10));
	texture14.bind(std::make_shared<cx::Image2D<unsigned char>>(allocator, 10, 10));
	texture14.bind(std::make_shared<cx::Image2D<unsigned short>>(allocator, 10, 10));

	stream.launch(test_device_texture, 1, 1)(texture0, texture1, texture2, texture3,
											 texture4, texture5, texture6, texture7,
											 texture8.accessor(), texture9.accessor(),
											 texture10, texture11, texture12, texture13);
	stream.sync();
}