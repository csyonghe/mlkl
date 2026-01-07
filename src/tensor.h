#pragma once

#include "../core/slang-basic.h"
#include "../core/slang-math.h"
#include "slang.h"
#include "stack-allocator.h"

#include <initializer_list>

struct Shape
{
    Slang::Array<int, 8> dims;

    Shape() = default;
    Shape(int i0) { dims.add(i0); }
    Shape(int i0, int i1)
    {
        dims.add(i0);
        dims.add(i1);
    }
    Shape(int i0, int i1, int i2)
    {
        dims.add(i0);
        dims.add(i1);
        dims.add(i2);
    }
    Shape(int i0, int i1, int i2, int i3)
    {
        dims.add(i0);
        dims.add(i1);
        dims.add(i2);
        dims.add(i3);
    }
    Shape(int i0, int i1, int i2, int i3, int i4)
    {
        dims.add(i0);
        dims.add(i1);
        dims.add(i2);
        dims.add(i3);
        dims.add(i4);
    }
    Shape(Slang::ArrayView<int> d)
    {
        for (auto v : d)
            dims.add(v);
    }
    Shape(const std::initializer_list<int>& d)
    {
        for (auto v : d)
            dims.add(v);
    }

    // Returns the tail shape starting from the given dimension.
    Shape tail(int startDim) const
    {
        Shape result;
        for (int i = startDim; i < getRank(); ++i)
            result.dims.add(dims[i]);
        return result;
    }

    // Returns the head shape up to the given dimension count.
    Shape head(int dimCount) const
    {
        Shape result;
        for (int i = 0; i < Slang::Math::Min(dimCount, getRank()); ++i)
            result.dims.add(dims[i]);
        return result;
    }

    bool isScalar() const { return dims.getCount() == 0; }
    int getRank() const { return (int)dims.getCount(); }
    int operator[](int i) const { return dims[i]; }
    Slang::ArrayView<int> getDims() const
    {
        return Slang::ArrayView<int>((int*)dims.getBuffer(), dims.getCount());
    }
    size_t getElementCount() const;

    bool operator==(const Shape& other) const;
    bool operator!=(const Shape& other) const { return !(*this == other); }
    bool isCompatibleWith(const Shape& other) const;
};

Slang::Array<uint32_t, 8> computeDenseStrides(const Shape& shape);

enum class ElementType
{
    Float16,
    Float32,
    Float8E4M3,
    Float8E5M2,
    Int8,
    Int16,
    Int32,
    UInt8,
    UInt16,
    UInt32,
};

size_t getElementTypeSize(ElementType elementType);

// Returns the Slang type name for the given element type.
// Used when generating shader type specializations.
inline const char* getSlangElementTypeName(ElementType elementType)
{
    switch (elementType)
    {
    case ElementType::Float32:
        return "float";
    case ElementType::Float16:
        return "half";
    case ElementType::Int32:
        return "int";
    // Future element types:
    // case ElementType::Float8E4M3:
    //     return "float8_e4m3";
    // case ElementType::Float8E5M2:
    //     return "float8_e5m2";
    default:
        throw std::runtime_error("Unsupported element type for Slang shader");
    }
}

// Convert float data to the specified element type.
// Returns a buffer containing the converted data.
// For Float32, the data is copied as-is.
// For Float16, the data is converted to half precision.
// For Int32, the data is converted by bitcast (reinterpret).
Slang::List<uint8_t> convertFloatData(const float* data, size_t count, ElementType targetType);

// Convert float List to the specified element type.
inline Slang::List<uint8_t> convertFloatData(const Slang::List<float>& data, ElementType targetType)
{
    return convertFloatData(data.getBuffer(), data.getCount(), targetType);
}

class Tensor;

class TensorView
{
public:
    BufferView bufferView;
    ElementType elementType = ElementType::Float32;
    Shape shape = {};
    TensorView() = default;
    TensorView(BufferView bufferView, ElementType elementType, Shape shape)
        : bufferView(bufferView), elementType(elementType), shape(shape)
    {
    }
    TensorView(const Tensor& tensor);
    operator bool() const { return bufferView.buffer != nullptr && bufferView.size > 0; }
    uint64_t getDeviceAddress() const { return bufferView.getDeviceAddress(); }
    size_t getElementSize() const { return getElementTypeSize(elementType); }
    size_t getBufferSize() const { return shape.getElementCount() * getElementSize(); }
    BufferView getBufferView() const { return bufferView; }
    TensorView reshape(const Shape& newShape) const
    {
        if (newShape.getElementCount() != shape.getElementCount())
        {
            throw std::runtime_error("TensorView::reshape: Element count mismatch");
        }
        return TensorView(bufferView, elementType, newShape);
    }

    // Return a slice of the tensor view along the first dimension.
    TensorView slice(int index, int count) const;

    // Ensure the tensor has the given rank, adding singleton dimensions if needed.
    TensorView ensureRank(int rank) const;
};

class Tensor : public Slang::RefObject
{
public:
    Slang::ComPtr<rhi::IBuffer> buffer;
    ElementType elementType = ElementType::Float32;
    Shape shape = {};
    Tensor() = default;
    size_t getElementSize() const { return getElementTypeSize(elementType); }
    size_t getBufferSize() const { return shape.getElementCount() * getElementSize(); }
    BufferView getBufferView() const { return BufferView(buffer.get(), 0, getBufferSize()); }
    TensorView getView() const { return TensorView(*this); }
};
