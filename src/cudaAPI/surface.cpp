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

#include "image.h"
#include "logger.h"
#include "surface.h"
#include <cuda_runtime_api.h>

CX_USING_NAMESPACE

/*********************************************************************************
*********************************    Surface    **********************************
*********************************************************************************/

Surface::Surface() : m_hSurface(0)
{

}


void Surface::bindImage(std::shared_ptr<Image> pImage)
{
	this->unbind();

	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = pImage->data().handle;

	cudaError_t err = cudaCreateSurfaceObject(&m_hSurface, &resDesc);

	CX_WARNING_LOG_IF(!pImage->isSurfaceLoadStoreSupported(), "Binding a image without 'bSurfaceLoadStore' flag.");

	if (err == cudaSuccess)
	{
		m_image = pImage;
	}
	else
	{
		CX_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}
}


void Surface::unbind()
{
	if (m_hSurface != 0)
	{
		cudaError_t err = cudaDestroySurfaceObject(m_hSurface);

		if (err != cudaSuccess)
		{
			CX_ERROR_LOG("%s.", cudaGetErrorString(err));

			cudaGetLastError();
		}

		m_image = nullptr;

		m_hSurface = 0;
	}
}


Surface::~Surface()
{
	this->unbind();
}