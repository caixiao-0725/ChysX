#include <cudaAPI/device.h>
#include <cudaAPI/context.h>
#include <cudaAPI/allocator.h>
#include <cudaAPI/device_host_vector.h>



void device_host_vector_test()
{
    cx::DeviceHostVector<int> vec;

    vec.Allocate(100);

    auto & hostVec = vec.GetHost();

    for(int i = 0; i < 100; i++){
        hostVec[i] = i;
    }

    vec.ReadToDevice();
    
}