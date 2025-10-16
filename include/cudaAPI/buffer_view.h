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
#include "buffer.h"
#include "device_pointer.h"

namespace CX_NAMESPACE
{
	/*****************************************************************************
	***************************    BufferView1D<T>    ****************************
	*****************************************************************************/

	//	A class representing a 1D view of a buffer.
	template<typename Type> class BufferView1D : public dev::Ptr<Type>
	{

	public:

		//	Default constructor.
		BufferView1D() : dev::Ptr<Type>(nullptr), m_buffer(nullptr), m_offset(0) {}

		//	Construct with nullptr.
		BufferView1D(std::nullptr_t) : dev::Ptr<Type>(nullptr), m_buffer(nullptr), m_offset(0) {}

		//	Copy constructor, initializes from another BufferView1D of the same type.
		BufferView1D(const BufferView1D<std::remove_cv_t<Type>> & rhs) : dev::Ptr<Type>(rhs.data(), rhs.width()), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		//	Construct with a given Buffer.
		explicit BufferView1D(std::shared_ptr<Buffer> buffer) : dev::Ptr<Type>(static_cast<Type*>(buffer->data()), buffer->capacity() / sizeof(Type)), m_buffer(buffer), m_offset(0) { CX_ASSERT(buffer != nullptr); }

		/**
		 *	@brief		Constructor to initialize with a shared buffer, offset and width.
		 *	@param[in]	buffer - Shared pointer to the buffer.
		 *	@param[in]	offset - offset within the buffer where the 1D view starts.
		 *	@param[in]	width - Width of the 1D view.
		 *	@warning	Ensures the buffer is not null, the address is correctly aligned, and the size fits within the buffer.
		 */
		explicit BufferView1D(std::shared_ptr<Buffer> buffer, size_t offset, size_t width) : dev::Ptr<Type>(reinterpret_cast<Type*>(buffer->address() + offset), width), m_buffer(buffer), m_offset(offset)
		{
			CX_ASSERT((buffer != nullptr));
			CX_ASSERT((buffer->address() + offset) % alignof(Type) == 0);
			CX_ASSERT((offset + sizeof(Type) * width) <= buffer->capacity());
		}

	public:

		//	Returns pointer to the associated Buffer.
		std::shared_ptr<Buffer> buffer() const { return m_buffer; }

		//	Returns device pointer to the underlying array, explicitly.
		const dev::Ptr<Type> & ptr() const { return *this; }

		//	Returns the offset within the buffer.
		size_t offset() const { return m_offset; }

	private:

		size_t						m_offset;
		std::shared_ptr<Buffer>		m_buffer;
	};

	/*****************************************************************************
	***************************    BufferView2D<T>    ****************************
	*****************************************************************************/

	//	A class representing a 2D view of a buffer.
	template<typename Type> class BufferView2D : public dev::Ptr2<Type>
	{

	public:

		//	Default constructor.
		BufferView2D() : dev::Ptr2<Type>(nullptr), m_buffer(nullptr), m_offset(0) {}

		//	Construct with nullptr.
		BufferView2D(std::nullptr_t) : dev::Ptr2<Type>(nullptr), m_buffer(nullptr), m_offset(0) {}

		//	Copy constructor, initializes from another BufferView2D of the same type.
		BufferView2D(const BufferView2D<std::remove_cv_t<Type>> & rhs) : dev::Ptr2<Type>(rhs.data(), rhs.width(), rhs.height()), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		//	Copy constructor, construct with a given BufferView1D.
		explicit BufferView2D(const BufferView1D<std::remove_cv_t<Type>> & rhs) : dev::Ptr2<Type>(rhs.data(), rhs.width(), 1), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		/**
		 *	@brief		Constructor to initialize with a shared buffer, offset, width, and height.
		 *	@param[in]	buffer - Shared pointer to the buffer.
		 *	@param[in]	offset - offset within the buffer where the 2D view starts.
		 *	@param[in]	width - Width of the 2D view.
		 *	@param[in]	height - Height of the 2D view.
		 *	@warning	Ensures the buffer is not null, the address is correctly aligned, and the size fits within the buffer.
		 */
		explicit BufferView2D(std::shared_ptr<Buffer> buffer, size_t offset, uint32_t width, uint32_t height) : dev::Ptr2<Type>(reinterpret_cast<Type*>(buffer->address() + offset), width, height), m_buffer(buffer), m_offset(offset)
		{
			CX_ASSERT((buffer != nullptr));
			CX_ASSERT((buffer->address() + offset) % alignof(Type) == 0);
			CX_ASSERT((offset + sizeof(Type) * width * height) <= buffer->capacity());
		}

	public:

		//	Returns pointer to the associated Buffer.
		std::shared_ptr<Buffer> buffer() const { return m_buffer; }

		//	Returns device pointer to the underlying array, explicitly.
		const dev::Ptr2<Type> & ptr() const { return *this; }

		//	Returns the offset within the buffer.
		size_t offset() const { return m_offset; }

	private:

		size_t						m_offset;
		std::shared_ptr<Buffer>		m_buffer;
	};

	/*****************************************************************************
	***************************    BufferView3D<T>    ****************************
	*****************************************************************************/

	//	A class representing a 3D view of a buffer.
	template<typename Type> class BufferView3D : public dev::Ptr3<Type>
	{

	public:

		//	Default constructor.
		BufferView3D() : dev::Ptr3<Type>(nullptr), m_buffer(nullptr), m_offset(0) {}

		//	Construct with nullptr.
		BufferView3D(std::nullptr_t) : dev::Ptr3<Type>(nullptr), m_buffer(nullptr), m_offset(0) {}

		//	Copy constructor, initializes from another BufferView3D of the same type.
		BufferView3D(const BufferView3D<std::remove_cv_t<Type>> & rhs) : dev::Ptr3<Type>(rhs.data(), rhs.width(), rhs.height(), rhs.depth()), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		//	Copy constructor, construct with a given BufferView2D.
		explicit BufferView3D(const BufferView2D<std::remove_cv_t<Type>> & rhs) : dev::Ptr3<Type>(rhs.data(), rhs.width(), rhs.height(), 1), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		//	Copy constructor, construct with a given BufferView1D.
		explicit BufferView3D(const BufferView1D<std::remove_cv_t<Type>> & rhs) : dev::Ptr3<Type>(rhs.data(), rhs.Width(), 1, 1), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		/**
		 *	@brief		Constructor to initialize with a shared buffer, offset, width, height and depth.
		 *	@param[in]	buffer - Shared pointer to the buffer.
		 *	@param[in]	offset - offset within the buffer where the 3D view starts.
		 *	@param[in]	width - Width of the 3D view.
		 *	@param[in]	height - Height of the 3D view.
		 *	@param[in]	depth - Depth of the 3D view.
		 *	@warning	Ensures the buffer is not null, the address is correctly aligned, and the size fits within the buffer.
		 */
		explicit BufferView3D(std::shared_ptr<Buffer> buffer, size_t offset, uint32_t width, uint32_t height, uint32_t depth) : dev::Ptr3<Type>(reinterpret_cast<Type*>(buffer->address() + offset), width, height, depth), m_buffer(buffer), m_offset(offset)
		{
			CX_ASSERT((buffer != nullptr));
			CX_ASSERT((buffer->address() + offset) % alignof(Type) == 0);
			CX_ASSERT((offset + sizeof(Type) * width * height * depth) <= buffer->capacity());
		}

	public:

		//	Returns pointer to the associated Buffer.
		std::shared_ptr<Buffer> buffer() const { return m_buffer; }

		//	Returns device pointer to the underlying array, explicitly.
		const dev::Ptr3<Type> & ptr() const { return *this; }

		//	Returns the offset within the buffer.
		size_t offset() const { return m_offset; }

	private:

		size_t						m_offset;
		std::shared_ptr<Buffer>		m_buffer;
	};

	/*****************************************************************************
	**************************    view_cast<T1, T2>    ***************************
	*****************************************************************************/

	//	Reinterpret a 1D buffer view as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferView1D<DstType> view_cast(BufferView1D<SrcType> view)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "view_cast requires DstType and SrcType to be binary compatible");

		return view.empty() ? BufferView1D<DstType>() : BufferView1D<DstType>(view.buffer(), view.offset(), view.width());
	}

	//	Reinterpret a 2D buffer view as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferView2D<DstType> view_cast(BufferView2D<SrcType> view)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "view_cast requires DstType and SrcType to be binary compatible");

		return view.empty() ? BufferView2D<DstType>() : BufferView2D<DstType>(view.buffer(), view.offset(), view.width(), view.height());
	}

	//	Reinterpret a 3D buffer view as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferView3D<DstType> view_cast(BufferView3D<SrcType> view)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "view_cast requires DstType and SrcType to be binary compatible");

		return view.empty() ? BufferView3D<DstType>() : BufferView3D<DstType>(view.buffer(), view.offset(), view.width(), view.height(), view.depth());
	}

	//	Reinterpret a 1D buffer view as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferView1D<const DstType> view_cast(BufferView1D<const SrcType> view)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "view_cast requires DstType and SrcType to be binary compatible");

		return view.empty() ? BufferView1D<const DstType>() : BufferView1D<const DstType>(view.buffer(), view.offset(), view.width());
	}

	//	Reinterpret a 2D buffer view as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferView2D<const DstType> view_cast(BufferView2D<const SrcType> view)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "view_cast requires DstType and SrcType to be binary compatible");

		return view.empty() ? BufferView2D<const DstType>() : BufferView2D<const DstType>(view.buffer(), view.offset(), view.width(), view.height());
	}

	//	Reinterpret a 3D buffer view as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferView3D<const DstType> view_cast(BufferView3D<const SrcType> view)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "view_cast requires DstType and SrcType to be binary compatible");

		return view.empty() ? BufferView3D<const DstType>() : BufferView3D<const DstType>(view.buffer(), view.offset(), view.width(), view.height(), view.depth());
	}
}