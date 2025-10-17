#include <cudaAPI/logger.h>
#include <cudaAPI/format.h>
#include <cudaAPI/device.h>
#include <cudaAPI/context.h>
#include <cudaAPI/allocator.h>

/*********************************************************************************
******************************    allocator_test    ******************************
*********************************************************************************/

class MyHostAllocator : public cx::HostAllocator
{
	virtual void * doAllocateMemory(size_t bytes) override
	{
		CX_INFO_LOG("Allocate host memory: %lld.", bytes);

		return cx::HostAllocator::doAllocateMemory(bytes);
	}
	virtual void doDeallocateMemory(void * ptr) override
	{
		cx::HostAllocator::doDeallocateMemory(ptr);

		CX_INFO_LOG("Deallocate host memory.");
	}
};


void allocator_test()
{
	auto device = cx::Context::getInstance()->device(0);

	MyHostAllocator hostAlloc;
	auto hostPtr = hostAlloc.allocateMemory(110);
	hostAlloc.deallocateMemory(hostPtr);

	cx::DeviceAllocator devAlloc(device);
	auto Ptr = devAlloc.allocateMemory(128);
	devAlloc.deallocateMemory(Ptr);

	auto pAlloc = device->defaultAllocator();
	auto texMem = pAlloc->allocateTextureMemory(cx::Format::Float, 100, 100, 100);
	devAlloc.deallocateTextureMemory(texMem);

	auto mipTexMem = pAlloc->allocateMipmapTextureMemory(cx::Format::Int, 100, 100, 100, 5);
	devAlloc.deallocateMipmapTextureMemory(mipTexMem);
}