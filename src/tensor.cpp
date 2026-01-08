#include "tensor.h"
#include "half.h"

using namespace Slang;

size_t getElementTypeSize(ElementType elementType)
{
    switch (elementType)
    {
    case ElementType::Float16:
        return 2;
    case ElementType::Float32:
        return 4;
    case ElementType::BFloat16:
        return 2;
    case ElementType::Float8E4M3:
        return 1;
    case ElementType::Float8E5M2:
        return 1;
    case ElementType::Int8:
        return 1;
    case ElementType::Int16:
        return 2;
    case ElementType::Int32:
        return 4;
    case ElementType::Int64:
        return 8;
    case ElementType::UInt8:
        return 1;
    case ElementType::UInt16:
        return 2;
    case ElementType::UInt32:
        return 4;
    case ElementType::UInt64:
        return 8;
    case ElementType::Bool:
        return 1;
    default:
        throw std::runtime_error("Unknown ElementType");
    }
}

size_t Shape::getElementCount() const
{
    if (isScalar())
        return 1;
    size_t count = 1;
    for (Index i = 0; i < dims.getCount(); i++)
        count *= dims[i];
    return count;
}
bool Shape::operator==(const Shape& other) const
{
    if (dims.getCount() != other.dims.getCount())
        return false;
    for (Index i = 0; i < dims.getCount(); i++)
        if (dims[i] != other.dims[i])
            return false;
    return true;
}
bool Shape::isCompatibleWith(const Shape& other) const
{
    if (this->isScalar() || other.isScalar())
        return true;
    return *this == other;
}

Array<uint32_t, 8> computeDenseStrides(const Shape& shape)
{
    Array<uint32_t, 8> strides;
    for (int i = 0; i < shape.getRank(); i++)
        strides.add(0);

    int s = 1;
    for (int i = shape.getRank() - 1; i >= 0; i--)
    {
        strides[i] = s;
        s *= shape[i];
    }
    return strides;
}

TensorView::TensorView(const Tensor& tensor)
{
    elementType = tensor.elementType;
    shape = tensor.shape;
    bufferView = tensor.getBufferView();
}

// Return a slice of the tensor view along the first dimension.
TensorView TensorView::slice(int index, int count) const
{
    if (shape.isScalar())
    {
        throw std::runtime_error("TensorView::slice: Cannot slice a scalar tensor");
    }
    if (index < 0 || count < 0 || index + count > shape.dims[0])
    {
        throw std::runtime_error("TensorView::slice: Slice indices out of bounds");
    }
    // Compute the size of each slice.
    size_t sliceElementCount = 1;
    for (Index i = 1; i < shape.getRank(); i++)
    {
        sliceElementCount *= shape.dims[i];
    }
    size_t sliceSizeInBytes = sliceElementCount * getElementSize();
    // Compute the new buffer view for the slice.
    BufferView slicedBufferView(
        bufferView.buffer,
        bufferView.offset + index * sliceSizeInBytes,
        count * sliceSizeInBytes);
    // Compute the new shape for the slice.
    Shape newShape;
    newShape.dims.add(count);
    for (Index i = 1; i < shape.getRank(); i++)
    {
        newShape.dims.add(shape.dims[i]);
    }
    return TensorView(slicedBufferView, elementType, newShape);
}

TensorView TensorView::ensureRank(int rank) const
{
    if (shape.getRank() >= rank)
    {
        return *this;
    }
    // Create new shape with singleton dimensions added to the front.
    Shape newShape;
    int dimsToAdd = rank - shape.getRank();
    for (int i = 0; i < dimsToAdd; i++)
    {
        newShape.dims.add(1);
    }
    for (Index i = 0; i < shape.getRank(); i++)
    {
        newShape.dims.add(shape.dims[i]);
    }
    return TensorView(bufferView, elementType, newShape);
}

List<uint8_t> convertFloatData(const float* data, size_t count, ElementType targetType)
{
    List<uint8_t> result;
    size_t elementSize = getElementTypeSize(targetType);
    result.setCount(count * elementSize);

    switch (targetType)
    {
    case ElementType::Float32:
        // Direct copy
        memcpy(result.getBuffer(), data, count * sizeof(float));
        break;

    case ElementType::Float16:
        {
            uint16_t* dst = (uint16_t*)result.getBuffer();
            for (size_t i = 0; i < count; i++)
            {
                dst[i] = floatToHalf(data[i]);
            }
            break;
        }

    case ElementType::Int32:
        {
            int32_t* dst = (int32_t*)result.getBuffer();
            for (size_t i = 0; i < count; i++)
            {
                dst[i] = (int32_t)data[i];
            }
            break;
        }

    default:
        throw std::runtime_error("convertFloatData: Unsupported target element type");
    }

    return result;
}