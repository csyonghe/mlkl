#include "safetensors-reader.h"

#include "core/slang-io.h"
#include "json.h"

// Convert half-precision float (F16) to single-precision float (F32)
static float halfToFloat(uint16_t h)
{
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0)
    {
        // Zero or denormalized
        if (mantissa == 0)
        {
            // Zero
            uint32_t result = sign;
            return *reinterpret_cast<float*>(&result);
        }
        else
        {
            // Denormalized - normalize it
            while ((mantissa & 0x400) == 0)
            {
                mantissa <<= 1;
                exponent--;
            }
            exponent++;
            mantissa &= 0x3FF;
        }
    }
    else if (exponent == 31)
    {
        // Infinity or NaN
        uint32_t result = sign | 0x7F800000 | (mantissa << 13);
        return *reinterpret_cast<float*>(&result);
    }

    // Normalized number
    exponent = exponent + (127 - 15);
    mantissa = mantissa << 13;

    uint32_t result = sign | (exponent << 23) | mantissa;
    return *reinterpret_cast<float*>(&result);
}

// IEEE 754 half-precision format conversion (F32 -> F16)
static uint16_t floatToHalf(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFF;

    if (exponent <= 0)
    {
        // Underflow to zero
        return (uint16_t)sign;
    }
    else if (exponent >= 31)
    {
        // Overflow to infinity
        return (uint16_t)(sign | 0x7C00);
    }

    return (uint16_t)(sign | (exponent << 10) | (mantissa >> 13));
}

// Convert SafeTensors dtype string to ElementType
static bool parseElementType(UnownedStringSlice dtypeStr, ElementType& outType)
{
    if (dtypeStr == toSlice("F32"))
    {
        outType = ElementType::Float32;
        return true;
    }
    else if (dtypeStr == toSlice("F16"))
    {
        outType = ElementType::Float16;
        return true;
    }
    else if (dtypeStr == toSlice("BF16"))
    {
        outType = ElementType::BFloat16;
        return true;
    }
    else if (dtypeStr == toSlice("I8"))
    {
        outType = ElementType::Int8;
        return true;
    }
    else if (dtypeStr == toSlice("I16"))
    {
        outType = ElementType::Int16;
        return true;
    }
    else if (dtypeStr == toSlice("I32"))
    {
        outType = ElementType::Int32;
        return true;
    }
    else if (dtypeStr == toSlice("I64"))
    {
        outType = ElementType::Int64;
        return true;
    }
    else if (dtypeStr == toSlice("U8"))
    {
        outType = ElementType::UInt8;
        return true;
    }
    else if (dtypeStr == toSlice("U16"))
    {
        outType = ElementType::UInt16;
        return true;
    }
    else if (dtypeStr == toSlice("U32"))
    {
        outType = ElementType::UInt32;
        return true;
    }
    else if (dtypeStr == toSlice("U64"))
    {
        outType = ElementType::UInt64;
        return true;
    }
    else if (dtypeStr == toSlice("BOOL"))
    {
        outType = ElementType::Bool;
        return true;
    }
    else if (dtypeStr == toSlice("F8_E4M3"))
    {
        outType = ElementType::Float8E4M3;
        return true;
    }
    else if (dtypeStr == toSlice("F8_E5M2"))
    {
        outType = ElementType::Float8E5M2;
        return true;
    }
    return false;
}

// Convert a single element from source type to float
static float elementToFloat(const void* src, ElementType srcType)
{
    switch (srcType)
    {
    case ElementType::Float32:
        return *reinterpret_cast<const float*>(src);
    case ElementType::Float16:
        return halfToFloat(*reinterpret_cast<const uint16_t*>(src));
    case ElementType::BFloat16:
        {
            uint32_t f32Bits = uint32_t(*reinterpret_cast<const uint16_t*>(src)) << 16;
            return *reinterpret_cast<const float*>(&f32Bits);
        }
    default:
        return 0.0f;
    }
}

// Convert a single float to target element type and write to destination
static void floatToElement(float value, void* dst, ElementType dstType)
{
    switch (dstType)
    {
    case ElementType::Float32:
        *reinterpret_cast<float*>(dst) = value;
        break;
    case ElementType::Float16:
        *reinterpret_cast<uint16_t*>(dst) = floatToHalf(value);
        break;
    case ElementType::BFloat16:
        {
            uint32_t f32Bits;
            memcpy(&f32Bits, &value, sizeof(float));
            *reinterpret_cast<uint16_t*>(dst) = (uint16_t)(f32Bits >> 16);
        }
        break;
    default:
        break;
    }
}

SlangResult SafeTensorsReader::load(const String& path)
{
    // 1. Memory-map the file (zero-copy)
    SLANG_RETURN_ON_FAIL(File::map(path, mappedFile));

    if (mappedFile.getSize() < 8)
    {
        return SLANG_FAIL;
    }

    const uint8_t* data = static_cast<const uint8_t*>(mappedFile.getData());

    // 2. Read header size (first 8 bytes, little-endian uint64)
    headerSize = 0;
    for (int i = 0; i < 8; i++)
    {
        headerSize |= (size_t(data[i]) << (i * 8));
    }

    if (8 + headerSize > mappedFile.getSize())
    {
        return SLANG_FAIL;
    }

    // 3. Parse JSON header using existing parseJson helper
    UnownedStringSlice headerJson((const char*)(data + 8), headerSize);
    auto parsedJson = parseJson(headerJson);
    if (!parsedJson)
    {
        return SLANG_FAIL;
    }

    if (parsedJson->rootValue.getKind() != JSONValue::Kind::Object)
    {
        return SLANG_FAIL;
    }

    // 4. Extract tensor metadata (parsedJson can be discarded after this)
    auto* container = parsedJson->container.get();
    auto objects = container->getObject(parsedJson->rootValue);
    for (const auto& kv : objects)
    {
        UnownedStringSlice name = container->getStringFromKey(kv.key);

        // Skip __metadata__
        if (name == toSlice("__metadata__"))
            continue;

        if (kv.value.getKind() != JSONValue::Kind::Object)
            continue;

        JSONValue info = kv.value;
        auto fields = container->getObject(info);

        SafeTensorInfo tensorInfo;

        for (const auto& field : fields)
        {
            UnownedStringSlice fieldName = container->getStringFromKey(field.key);

            if (fieldName == toSlice("dtype"))
            {
                UnownedStringSlice dtypeStr = container->getString(field.value);
                if (!parseElementType(dtypeStr, tensorInfo.dtype))
                {
                    // Unknown dtype - skip this tensor or use a default
                    tensorInfo.dtype = ElementType::Float32;
                }
            }
            else if (fieldName == toSlice("shape"))
            {
                if (field.value.getKind() == JSONValue::Kind::Array)
                {
                    auto shapeArray = container->getArray(field.value);
                    for (const auto& dim : shapeArray)
                    {
                        tensorInfo.shape.dims.add((int)container->asInteger(dim));
                    }
                }
            }
            else if (fieldName == toSlice("data_offsets"))
            {
                if (field.value.getKind() == JSONValue::Kind::Array)
                {
                    auto offsets = container->getArray(field.value);
                    if (offsets.getCount() >= 2)
                    {
                        int64_t start = container->asInteger(offsets[0]);
                        int64_t end = container->asInteger(offsets[1]);
                        tensorInfo.dataOffset = size_t(start);
                        tensorInfo.dataSize = size_t(end - start);
                    }
                }
            }
        }

        tensors[String(name)] = tensorInfo;
    }

    return SLANG_OK;
}

bool SafeTensorsReader::hasTensor(UnownedStringSlice name) const
{
    return tensors.containsKey(String(name));
}

const SafeTensorInfo* SafeTensorsReader::getTensorInfo(UnownedStringSlice name) const
{
    const SafeTensorInfo* result = tensors.tryGetValue(String(name));
    return result;
}

const void* SafeTensorsReader::getTensorData(UnownedStringSlice name) const
{
    const SafeTensorInfo* info = getTensorInfo(name);
    if (!info)
        return nullptr;
    return getDataSection() + info->dataOffset;
}

void SafeTensorsReader::getTensorNames(List<String>& outNames) const
{
    outNames.clear();
    for (const auto& kv : tensors)
    {
        outNames.add(kv.first);
    }
}

SlangResult SafeTensorsReader::readTensor(
    UnownedStringSlice name,
    ElementType targetType,
    List<uint8_t>& outData,
    ConstArrayView<int> permutation) const
{
    const SafeTensorInfo* info = getTensorInfo(name);
    if (!info)
    {
        printf(
            "warning: tensor '%.*s' not found in weights file\n",
            (int)name.getLength(),
            name.begin());
        return SLANG_E_NOT_FOUND;
    }

    size_t numElements = info->shape.getElementCount();
    size_t srcElementSize = getElementTypeSize(info->dtype);
    size_t dstElementSize = getElementTypeSize(targetType);

    outData.setCount(numElements * dstElementSize);

    const uint8_t* src = getDataSection() + info->dataOffset;
    uint8_t* dst = outData.getBuffer();

    // No permutation case
    if (permutation.getCount() == 0)
    {
        // Fast path: same type, just copy
        if (info->dtype == targetType)
        {
            memcpy(dst, src, numElements * srcElementSize);
            return SLANG_OK;
        }

        // Conversion path: convert through float element by element
        for (size_t i = 0; i < numElements; i++)
        {
            float value = elementToFloat(src + i * srcElementSize, info->dtype);
            floatToElement(value, dst + i * dstElementSize, targetType);
        }

        return SLANG_OK;
    }

    // Permutation case
    int rank = info->shape.getRank();
    if (permutation.getCount() != rank)
    {
        return SLANG_E_INVALID_ARG;
    }

    // Compute source strides (row-major)
    int srcStrides[8];
    srcStrides[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; d--)
    {
        srcStrides[d] = srcStrides[d + 1] * info->shape[d + 1];
    }

    // Compute destination shape and strides after permutation
    int dstShape[8];
    int dstStrides[8];
    for (int d = 0; d < rank; d++)
    {
        dstShape[d] = info->shape[permutation[d]];
    }
    dstStrides[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; d--)
    {
        dstStrides[d] = dstStrides[d + 1] * dstShape[d + 1];
    }

    // Iterate through all elements
    // We iterate in destination order for cache-friendly writes
    int coords[8] = {0};

    for (size_t dstIdx = 0; dstIdx < numElements; dstIdx++)
    {
        // Compute source index from destination coordinates via inverse permutation
        size_t srcIdx = 0;
        for (int d = 0; d < rank; d++)
        {
            // coords[d] is the coordinate in destination dimension d
            // This corresponds to source dimension permutation[d]
            srcIdx += coords[d] * srcStrides[permutation[d]];
        }

        // Read, convert, write
        float value = elementToFloat(src + srcIdx * srcElementSize, info->dtype);
        floatToElement(value, dst + dstIdx * dstElementSize, targetType);

        // Increment coordinates (like a multi-digit counter)
        for (int d = rank - 1; d >= 0; d--)
        {
            coords[d]++;
            if (coords[d] < dstShape[d])
                break;
            coords[d] = 0;
        }
    }

    return SLANG_OK;
}
