#pragma once

#include <memory>
#include "macros.h"

/*********************************************************************************
***************************    Forward Declarations    ***************************
*********************************************************************************/

#ifndef __CUDACC__
	struct dim3;
	struct cudaDeviceProp;
	typedef enum cudaError : int				cudaError_t;
	typedef struct cudaArray *					cudaArray_t;
	typedef struct CUevent_st *					cudaEvent_t;
	typedef struct CUgraph_st *					cudaGraph_t;
	typedef struct CUstream_st *				cudaStream_t;
	typedef struct CUgraphNode_st *				cudaGraphNode_t;
	typedef struct CUgraphExec_st *				cudaGraphExec_t;
	typedef struct cudaMipmappedArray *			cudaMipmappedArray_t;
	typedef unsigned long long					cudaTextureObject_t;
	typedef unsigned long long					cudaSurfaceObject_t;
#endif

namespace CX_NAMESPACE
{
	class Event;
	class Graph;
	class Buffer;
	class Stream;
	class Device;
	class Context;
	class Texture;
	class Allocator;
	class TimedEvent;
	class ScopedTimer;
	class HostAllocator;
	class DeviceAllocator;

	struct Version;
	struct Sampler;
	enum class Format;
	enum class Result;
	enum class FilterMode;
	enum class AddressMode;

	//	For device objects.
	namespace dev
	{
		template<typename Type> struct Ptr;
		template<typename Type> struct Ptr2;
		template<typename Type> struct Ptr3;

		template<typename Type> struct Surf1D;
		template<typename Type> struct Surf2D;
		template<typename Type> struct Surf3D;
		template<typename Type> struct SurfCube;
		template<typename Type> struct Surf1DLayered;
		template<typename Type> struct Surf2DLayered;
		template<typename Type> struct SurfCubeLayered;

		template<typename Type> struct Tex1D;
		template<typename Type> struct Tex2D;
		template<typename Type> struct Tex3D;
		template<typename Type> struct TexCube;
		template<typename Type> struct Tex1DLod;
		template<typename Type> struct Tex2DLod;
		template<typename Type> struct Tex3DLod;
		template<typename Type> struct TexCubeLod;
		template<typename Type> struct Tex1DLayered;
		template<typename Type> struct Tex2DLayered;
		template<typename Type> struct TexCubeLayered;
		template<typename Type> struct Tex1DLayeredLod;
		template<typename Type> struct Tex2DLayeredLod;
		template<typename Type> struct TexCubeLayeredLod;
	}

	template<typename Type> class Array;
	template<typename Type> class Array2D;
	template<typename Type> class Array3D;

	template<typename Type> class BufferView1D;
	template<typename Type> class BufferView2D;
	template<typename Type> class BufferView3D;

	template<typename Type> class Image1D;
	template<typename Type> class Image2D;
	template<typename Type> class Image3D;
	template<typename Type> class ImageCube;
	template<typename Type> class Image1DLayered;
	template<typename Type> class Image2DLayered;
	template<typename Type> class ImageCubeLayered;
	template<typename Type> class Image1DLod;
	template<typename Type> class Image2DLod;
	template<typename Type> class Image3DLod;
	template<typename Type> class ImageCubeLod;
	template<typename Type> class Image1DLayeredLod;
	template<typename Type> class Image2DLayeredLod;
	template<typename Type> class ImageCubeLayeredLod;

	template<typename Type> class Surface1D;
	template<typename Type> class Surface2D;
	template<typename Type> class Surface3D;
	template<typename Type> class SurfaceCube;
	template<typename Type> class Surface1DLayered;
	template<typename Type> class Surface2DLayered;
	template<typename Type> class SurfaceCubeLayered;

	template<typename Type> class Texture1D;
	template<typename Type> class Texture2D;
	template<typename Type> class Texture3D;
	template<typename Type> class TextureCube;
	template<typename Type> class Texture1DLod;
	template<typename Type> class Texture2DLod;
	template<typename Type> class Texture3DLod;
	template<typename Type> class TextureCubeLod;
	template<typename Type> class Texture1DLayered;
	template<typename Type> class Texture2DLayered;
	template<typename Type> class TextureCubeLayered;
	template<typename Type> class Texture1DLayeredLod;
	template<typename Type> class Texture2DLayeredLod;
	template<typename Type> class TextureCubeLayeredLod;

	template<typename Type> struct ImageAccessor;
	template<typename... Args> using KernelFunc = void(*)(Args...);

	using BufferPtr			= std::shared_ptr<Buffer>;
	using AllocPtr			= std::shared_ptr<Allocator>;
	using HostAllocPtr		= std::shared_ptr<HostAllocator>;
	using DevAllocPtr		= std::shared_ptr<DeviceAllocator>;

	//	Trait to check if two types are binary compatible in terms of size and alignment.
	template<typename Type1, typename Type2> struct BinaryCompatible
	{
		static constexpr bool value = (sizeof(Type1) == sizeof(Type2)) && (alignof(Type1) == alignof(Type2));
	};

	//	Utility functions to reinterpret buffer views as another compatible element type.
	template<typename DstType, typename SrcType> BufferView1D<DstType> view_cast(BufferView1D<SrcType> view);
	template<typename DstType, typename SrcType> BufferView2D<DstType> view_cast(BufferView2D<SrcType> view);
	template<typename DstType, typename SrcType> BufferView3D<DstType> view_cast(BufferView3D<SrcType> view);

	template<typename DstType, typename SrcType> BufferView1D<const DstType> view_cast(BufferView1D<const SrcType> view);
	template<typename DstType, typename SrcType> BufferView2D<const DstType> view_cast(BufferView2D<const SrcType> view);
	template<typename DstType, typename SrcType> BufferView3D<const DstType> view_cast(BufferView3D<const SrcType> view);

	// Utility functions to reinterpret device pointers as another compatible element type.
	template<typename DstType, typename SrcType> CX_CUDA_CALLABLE dev::Ptr<const DstType> ptr_cast(dev::Ptr<const SrcType> ptr);
	template<typename DstType, typename SrcType> CX_CUDA_CALLABLE dev::Ptr2<const DstType> ptr_cast(dev::Ptr2<const SrcType> ptr);
	template<typename DstType, typename SrcType> CX_CUDA_CALLABLE dev::Ptr3<const DstType> ptr_cast(dev::Ptr3<const SrcType> ptr);

	template<typename DstType, typename SrcType> CX_CUDA_CALLABLE dev::Ptr<DstType> ptr_cast(dev::Ptr<SrcType> ptr);
	template<typename DstType, typename SrcType> CX_CUDA_CALLABLE dev::Ptr2<DstType> ptr_cast(dev::Ptr2<SrcType> ptr);
	template<typename DstType, typename SrcType> CX_CUDA_CALLABLE dev::Ptr3<DstType> ptr_cast(dev::Ptr3<SrcType> ptr);
}

/*********************************************************************************
********************************    Type Alias    ********************************
*********************************************************************************/

using NsEvent											= CX_NAMESPACE::Event;
using NsGraph											= CX_NAMESPACE::Graph;
using NsBuffer											= CX_NAMESPACE::Buffer;
using NsStream											= CX_NAMESPACE::Stream;
using NsDevice											= CX_NAMESPACE::Device;
using NsFormat											= CX_NAMESPACE::Format;
using NsSampler											= CX_NAMESPACE::Sampler;
using NsContext											= CX_NAMESPACE::Context;
using NsVersion											= CX_NAMESPACE::Version;
using NsFilterMode										= CX_NAMESPACE::FilterMode;
using NsTimedEvent										= CX_NAMESPACE::TimedEvent;
using NsScopedTimer										= CX_NAMESPACE::ScopedTimer;
using NsAddressMode										= CX_NAMESPACE::AddressMode;
using NsHostAlloc										= CX_NAMESPACE::HostAllocator;
using NsDevAlloc										= CX_NAMESPACE::DeviceAllocator;
using NsAlloc											= CX_NAMESPACE::Allocator;

using NsBufferPtr										= CX_NAMESPACE::BufferPtr;
using NsHostAllocPtr									= CX_NAMESPACE::HostAllocPtr;
using NsDevAllocPtr										= CX_NAMESPACE::DevAllocPtr;
using NsAllocPtr										= CX_NAMESPACE::AllocPtr;

template<typename Type> using NsArray					= CX_NAMESPACE::Array<Type>;
template<typename Type> using NsArray2D					= CX_NAMESPACE::Array2D<Type>;
template<typename Type> using NsArray3D					= CX_NAMESPACE::Array3D<Type>;

template<typename Type> using NsBufferView1D			= CX_NAMESPACE::BufferView1D<Type>;
template<typename Type> using NsBufferView2D			= CX_NAMESPACE::BufferView2D<Type>;
template<typename Type> using NsBufferView3D			= CX_NAMESPACE::BufferView3D<Type>;

template<typename Type> using NsImage1D					= CX_NAMESPACE::Image1D<Type>;
template<typename Type> using NsImage2D					= CX_NAMESPACE::Image2D<Type>;
template<typename Type> using NsImage3D					= CX_NAMESPACE::Image3D<Type>;
template<typename Type> using NsImageCube				= CX_NAMESPACE::ImageCube<Type>;
template<typename Type> using NsImage1DLayered			= CX_NAMESPACE::Image1DLayered<Type>;
template<typename Type> using NsImage2DLayered			= CX_NAMESPACE::Image2DLayered<Type>;
template<typename Type> using NsImageCubeLayered		= CX_NAMESPACE::ImageCubeLayered<Type>;

template<typename Type> using NsImage1DLod				= CX_NAMESPACE::Image1DLod<Type>;
template<typename Type> using NsImage2DLod				= CX_NAMESPACE::Image2DLod<Type>;
template<typename Type> using NsImage3DLod				= CX_NAMESPACE::Image3DLod<Type>;
template<typename Type> using NsImageCubeLod			= CX_NAMESPACE::ImageCubeLod<Type>;
template<typename Type> using NsImage1DLayeredLod		= CX_NAMESPACE::Image1DLayeredLod<Type>;
template<typename Type> using NsImage2DLayeredLod		= CX_NAMESPACE::Image2DLayeredLod<Type>;
template<typename Type> using NsImageCubeLayeredLod		= CX_NAMESPACE::ImageCubeLayeredLod<Type>;

template<typename Type> using NsSurf1D					= CX_NAMESPACE::Surface1D<Type>;
template<typename Type> using NsSurf2D					= CX_NAMESPACE::Surface2D<Type>;
template<typename Type> using NsSurf3D					= CX_NAMESPACE::Surface3D<Type>;
template<typename Type> using NsSurfCube				= CX_NAMESPACE::SurfaceCube<Type>;
template<typename Type> using NsSurf1DLayered			= CX_NAMESPACE::Surface1DLayered<Type>;
template<typename Type> using NsSurf2DLayered			= CX_NAMESPACE::Surface2DLayered<Type>;
template<typename Type> using NsSurfCubeLayered			= CX_NAMESPACE::SurfaceCubeLayered<Type>;

template<typename Type> using NsTex1D					= CX_NAMESPACE::Texture1D<Type>;
template<typename Type> using NsTex2D					= CX_NAMESPACE::Texture2D<Type>;
template<typename Type> using NsTex3D					= CX_NAMESPACE::Texture3D<Type>;
template<typename Type> using NsTexCube					= CX_NAMESPACE::TextureCube<Type>;
template<typename Type> using NsTex1DLod				= CX_NAMESPACE::Texture1DLod<Type>;
template<typename Type> using NsTex2DLod				= CX_NAMESPACE::Texture2DLod<Type>;
template<typename Type> using NsTex3DLod				= CX_NAMESPACE::Texture3DLod<Type>;
template<typename Type> using NsTexCubeLod				= CX_NAMESPACE::TextureCubeLod<Type>;
template<typename Type> using NsTex1DLayered			= CX_NAMESPACE::Texture1DLayered<Type>;
template<typename Type> using NsTex2DLayered			= CX_NAMESPACE::Texture2DLayered<Type>;
template<typename Type> using NsTexCubeLayered			= CX_NAMESPACE::TextureCubeLayered<Type>;
template<typename Type> using NsTex1DLayeredLod			= CX_NAMESPACE::Texture1DLayeredLod<Type>;
template<typename Type> using NsTex2DLayeredLod			= CX_NAMESPACE::Texture2DLayeredLod<Type>;
template<typename Type> using NsTexCubeLayeredLod		= CX_NAMESPACE::TextureCubeLayeredLod<Type>;