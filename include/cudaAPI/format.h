#pragma once

#include "fwd.h"
#include "vector_types.h"

namespace CX_NAMESPACE
{
	/*****************************************************************************
	********************************    Format    ********************************
	*****************************************************************************/

	/**
	 *	@brief	Available texel formats for CUDA texture objects.
	 *	@note	This enum class defines the supported data types and component counts
	 *			for texels in CUDA texture objects. Each format specifies a base type
	 *			(e.g., char, int, float) and the number of components (1, 2, or 4).
	 */
	enum class Format
	{
		Int, Int2, Int4,			// Signed 32-bit integer (1, 2, or 4 components)
		Uint, Uint2, Uint4,			// Unsigned 32-bit integer (1, 2, or 4 components)
		Char, Char2, Char4,			// Signed 8-bit integer (1, 2, or 4 components)
		Uchar, Uchar2, Uchar4,		// Unsigned 8-bit integer (1, 2, or 4 components)
		Short, Short2, Short4,		// Signed 16-bit integer (1, 2, or 4 components)
		Ushort, Ushort2, Ushort4,	// Unsigned 16-bit integer (1, 2, or 4 components)
		Float, Float2, Float4,		// 32-bit floating-point (1, 2, or 4 components)
	};

	/*****************************************************************************
	*********************    FormatTraits / FormatMapping    *********************
	*****************************************************************************/

	namespace details
	{
	#ifdef __CUDACC__		//	skip stupid nvcc bug
		template<Format format> struct FormatTraits;
		template<typename Type> struct FormatMapping;
	#else
		template<Format format> struct FormatTraits	 { static_assert(false, "Invalid format!"); };
		template<typename Type> struct FormatMapping { static_assert(false, "No FormatMapping specialization found for this type!"); };
	#endif

		//	Maps C++ types to corresponding Format enum values.
		template<> struct FormatMapping<int>  { static constexpr Format value = Format::Int; };
		template<> struct FormatMapping<int2> { static constexpr Format value = Format::Int2; };
		template<> struct FormatMapping<int4> { static constexpr Format value = Format::Int4; };

		template<> struct FormatMapping<uint>  { static constexpr Format value = Format::Uint; };
		template<> struct FormatMapping<uint2> { static constexpr Format value = Format::Uint2; };
		template<> struct FormatMapping<uint4> { static constexpr Format value = Format::Uint4; };

		template<> struct FormatMapping<char>  { static constexpr Format value = Format::Char; };
		template<> struct FormatMapping<char2> { static constexpr Format value = Format::Char2; };
		template<> struct FormatMapping<char4> { static constexpr Format value = Format::Char4; };

		template<> struct FormatMapping<uchar>  { static constexpr Format value = Format::Uchar; };
		template<> struct FormatMapping<uchar2> { static constexpr Format value = Format::Uchar2; };
		template<> struct FormatMapping<uchar4> { static constexpr Format value = Format::Uchar4; };

		template<> struct FormatMapping<short>  { static constexpr Format value = Format::Short; };
		template<> struct FormatMapping<short2> { static constexpr Format value = Format::Short2; };
		template<> struct FormatMapping<short4> { static constexpr Format value = Format::Short4; };

		template<> struct FormatMapping<ushort>  { static constexpr Format value = Format::Ushort; };
		template<> struct FormatMapping<ushort2> { static constexpr Format value = Format::Ushort2; };
		template<> struct FormatMapping<ushort4> { static constexpr Format value = Format::Ushort4; };

		template<> struct FormatMapping<float>  { static constexpr Format value = Format::Float; };
		template<> struct FormatMapping<float2> { static constexpr Format value = Format::Float2; };
		template<> struct FormatMapping<float4> { static constexpr Format value = Format::Float4; };

		//	Type traits for Format
		template<> struct FormatTraits<Format::Int>		{ using type = int;		using component_type = int;		static constexpr int component_count = 1; };
		template<> struct FormatTraits<Format::Int2>	{ using type = int2;	using component_type = int;		static constexpr int component_count = 2; };
		template<> struct FormatTraits<Format::Int4>	{ using type = int4;	using component_type = int;		static constexpr int component_count = 4; };

		template<> struct FormatTraits<Format::Uint>	{ using type = uint;	using component_type = uint;	static constexpr int component_count = 1; };
		template<> struct FormatTraits<Format::Uint2>	{ using type = uint2;	using component_type = uint;	static constexpr int component_count = 2; };
		template<> struct FormatTraits<Format::Uint4>	{ using type = uint4;	using component_type = uint;	static constexpr int component_count = 4; };

		template<> struct FormatTraits<Format::Char>	{ using type = char;	using component_type = char;	static constexpr int component_count = 1; };
		template<> struct FormatTraits<Format::Char2>	{ using type = char2;	using component_type = char;	static constexpr int component_count = 2; };
		template<> struct FormatTraits<Format::Char4>	{ using type = char4;	using component_type = char;	static constexpr int component_count = 4; };

		template<> struct FormatTraits<Format::Uchar>	{ using type = uchar;	using component_type = uchar;	static constexpr int component_count = 1; };
		template<> struct FormatTraits<Format::Uchar2>	{ using type = uchar2;	using component_type = uchar;	static constexpr int component_count = 2; };
		template<> struct FormatTraits<Format::Uchar4>	{ using type = uchar4;	using component_type = uchar;	static constexpr int component_count = 4; };

		template<> struct FormatTraits<Format::Short>	{ using type = short;	using component_type = short;	static constexpr int component_count = 1; };
		template<> struct FormatTraits<Format::Short2>	{ using type = short2;	using component_type = short;	static constexpr int component_count = 2; };
		template<> struct FormatTraits<Format::Short4>	{ using type = short4;	using component_type = short;	static constexpr int component_count = 4; };

		template<> struct FormatTraits<Format::Ushort>	{ using type = ushort;	using component_type = ushort;	static constexpr int component_count = 1; };
		template<> struct FormatTraits<Format::Ushort2>	{ using type = ushort2;	using component_type = ushort;	static constexpr int component_count = 2; };
		template<> struct FormatTraits<Format::Ushort4>	{ using type = ushort4;	using component_type = ushort;	static constexpr int component_count = 4; };

		template<> struct FormatTraits<Format::Float>	{ using type = float;	using component_type = float;	static constexpr int component_count = 1; };
		template<> struct FormatTraits<Format::Float2>	{ using type = float2;	using component_type = float;	static constexpr int component_count = 2; };
		template<> struct FormatTraits<Format::Float4>	{ using type = float4;	using component_type = float;	static constexpr int component_count = 4; };
	}

	/*****************************************************************************
	****************************    FormatMapping    *****************************
	*****************************************************************************/

	/**
	 *	@brief	Wrapper for format mapping, defaults to FormatMapping.
	 *	@note	Allows customization of format mappings while providing default mappings
	 *			for standard types. Used to associate types with Format enum values.
	 */
	template<typename Type> struct FormatMapping
	{
		static constexpr Format value = details::FormatMapping<Type>::value;
	};

	//	Defines the internal CUDA-compatible value type for a given C++ type.
	template<typename Type> using InternalValueType = typename details::FormatTraits<FormatMapping<std::remove_const_t<Type>>::value>::type;

	/*****************************************************************************
	*************************    IsValidFormatMapping    *************************
	*****************************************************************************/

	/**
	 *	@brief	Validates type mappings at compile-time for CUDA texture configurations.
	 *	@note	Uses static assertions to ensure that the size and alignment of Type match
	 *			those of the type mapped from its corresponding Format enum value. This
	 *			ensures type safety and consistency in CUDA texture object configurations.
	 */
	template<typename Type> struct CheckFormatMapping
	{
		using underlying_type = typename details::FormatTraits<FormatMapping<Type>::value>::type;

		static_assert(sizeof(Type) == sizeof(underlying_type), "Size mismatch in format mapping!");
		static_assert(alignof(Type) == alignof(underlying_type), "Alignment mismatch in format mapping!");
	};
}