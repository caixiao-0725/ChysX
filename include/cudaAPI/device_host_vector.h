#pragma once

#include "array_1d.h"
#include <vector>
#include "helper_cuda.h"
#include "cuda_runtime.h"

namespace CX_NAMESPACE
{
    template<typename T>
    class DeviceHostVector
    {
        CX_NONCOPYABLE(DeviceHostVector)

    private:
		// Device memory
		cx::Array<T> deviceVector;

		// Host memory
		std::vector<T> hostVector;

		// Allocated memory size
		size_t size = 0;

    public:
        // Copy-swap implementation
        // Swap function of this class
        // Should be updated if you add new member variables
        friend void swap(DeviceHostVector<T>& m1, DeviceHostVector<T>& m2) noexcept
        {
            std::swap(m1.deviceVector, m2.deviceVector); // Swap resource pointer to avoid multi-release
            std::swap(m1.size, m2.size);
            std::swap(m1.hostVector, m2.hostVector);
        };


        // Default constructor
        DeviceHostVector() = default;
        
        // Free memory and reset members
		~DeviceHostVector() {}

        // Custom move constructor
        DeviceHostVector(DeviceHostVector&& other) noexcept { swap(*this, other); }

        // Custom move assignment operator
        DeviceHostVector& operator=(DeviceHostVector&& other) noexcept
        {
            swap(*this, other);
            return *this;
        }

        // Clear member without release memory pointer
		void Reset();

        // Get host memory vector
		std::vector<T> &GetHost() { return hostVector; }
        const std::vector<T> &GetHost() const { return hostVector; }
        
        // Get device memory vector
        cx::Array<T> &GetDevice() { return deviceVector; }
        const cx::Array<T> &GetDevice() const { return deviceVector; }

        // Get size
		size_t GetSize() const { return size; }

        // CPU mem set
		void SetHost(size_t newSize, T value){
            std::fill(hostVector.begin(), hostVector.begin() + newSize, value);
            size = newSize;
        }

		// CPU mem set
		void SetHost(size_t newSize, const T *hostData){
            std::copy(hostData, hostData + newSize, hostVector.begin());
            size = newSize;
        }

        // CPU mem set
		void SetHost(std::vector<T> &hostData){
            std::copy(hostData.begin(), hostData.end(), hostVector.begin());
            size = hostData.size();
        }

        // Copy host to device memory
		void ReadToDevice() {
            if(deviceVector.size() != size){
                deviceVector.resize(size);
            }
            checkCudaErrors(cudaMemcpy(deviceVector.data(), hostVector.data(), size * sizeof(T), cudaMemcpyHostToDevice));
        }

		// Copy device to host memory
		void ReadToHost() {
            if(hostVector.size() != size){
                hostVector.resize(size);
            }
            checkCudaErrors(cudaMemcpy(hostVector.data(), deviceVector.data(), size * sizeof(T), cudaMemcpyDeviceToHost));
        }

        // Allocate memory
        void Allocate(size_t newSize){
            if(newSize > size){
                size = newSize;
                deviceVector.resize(size);
                hostVector.resize(size);
            }
        }

        // Allocate host memory
        void AllocateHost(size_t newSize){
            if(newSize > size){
                size = newSize;
                hostVector.resize(size);
            }
        }

        // Allocate device memory
        void AllocateDevice(size_t newSize){
            if(newSize > size){
                size = newSize;
                deviceVector.resize(size);
            }
        }


    };
}