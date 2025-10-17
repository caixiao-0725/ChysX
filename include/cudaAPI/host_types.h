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
#pragma once

#include "fwd.h"
#include "vector_types.h"

namespace CX_NAMESPACE
{
	/*****************************************************************************
	*******************************    Version    ********************************
	*****************************************************************************/

	/**
	 *	@brief		CUDA version number.
	 */
	struct Version
	{
		union
		{
			struct { int Minor, Major; };
			struct { long long Encoded; };
		};

		//	Constructors
		constexpr Version() : Major(0), Minor(0) {}
		constexpr Version(int major, int minor) : Major(major), Minor(minor) {}

		//	Compare operators
		constexpr bool operator==(Version rhs) const { return Encoded == rhs.Encoded; }
		constexpr bool operator!=(Version rhs) const { return Encoded != rhs.Encoded; }
		constexpr bool operator<=(Version rhs) const { return Encoded <= rhs.Encoded; }
		constexpr bool operator>=(Version rhs) const { return Encoded >= rhs.Encoded; }
		constexpr bool operator<(Version rhs) const { return Encoded < rhs.Encoded; }
		constexpr bool operator>(Version rhs) const { return Encoded > rhs.Encoded; }
	};

	/*****************************************************************************
	****************************    ImageAccessor    *****************************
	*****************************************************************************/

	/**
	 *	@brief		A generic image accessor template for CUDA arrays.
	 *	@note		This provides position-aware access to CUDA array memory with support for
	 *				1D, 2D and 3D offset operations. The void specialization serves as the base
	 *				implementation containing common functionality.
	 */
	template<> struct ImageAccessor<void>
	{
		//	CUDA array handle (device memory pointer)
		cudaArray_t handle = nullptr;
		
		//	Current 3D position in the array (x, y, z coordinates)
		ulonglong3 pos = { 0, 0, 0 };

	public:

		//	Creates a new accessor with 1D offset applied.
		ImageAccessor operator+(ulonglong offset) const
		{
			return ImageAccessor{ handle, ulonglong3{ pos.x + offset, pos.y, pos.z } };
		}

		//	Creates a new accessor with 2D offset applied.
		ImageAccessor operator+(ulonglong2 offset) const
		{
			return ImageAccessor{ handle, ulonglong3{ pos.x + offset.x, pos.y + offset.y, pos.z } };
		}

		//	Creates a new accessor with 3D offset applied.
		ImageAccessor operator+(ulonglong3 offset) const
		{
			return ImageAccessor{ handle, ulonglong3{ pos.x + offset.x, pos.y + offset.y, pos.z + offset.z } };
		}

		//	Implicit conversion to underlying CUDA array handle
		operator cudaArray_t() const { return handle; }
	};


	/**
	 *	@brief		Typed image accessor inheriting base functionality
	 *	@tparam		Type - The pixel/element type of the CUDA array
	 *	@note		This template specialization adds type safety while maintaining
	 *				all positional access capabilities of the base class.
	 */
	template<typename Type> struct ImageAccessor : public ImageAccessor<void> {};
}