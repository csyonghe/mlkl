#pragma once

#include "core/slang-basic.h"
#include "core/slang-io.h"
#include "tensor.h"

using namespace Slang;

// SafeTensors Reader for loading model weights
//
// SafeTensors is a safe, fast format for storing tensors (https://huggingface.co/docs/safetensors)
// This reader provides zero-copy memory-mapped access to tensor data.
//
// USAGE:
//   SafeTensorsReader reader;
//   reader.load("model.safetensors");
//   
//   // Check available tensors
//   List<String> names;
//   reader.getTensorNames(names);  // Useful for debugging weight names
//   
//   // Read with optional permutation and type conversion
//   List<uint8_t> data;
//   reader.readTensor("layer.weight", ElementType::Float32, data, {1,2,3,0});
//
// TIPS:
// - Use getTensorNames() to discover exact weight names in a model file
// - Weight naming conventions vary by model architecture

// Tensor metadata from SafeTensors header
struct SafeTensorInfo
{
    ElementType dtype = ElementType::Float32;
    Shape shape;
    size_t dataOffset = 0;         // Offset from start of data section
    size_t dataSize = 0;           // Size in bytes
};

class SafeTensorsReader : public RefObject
{
private:
    Dictionary<String, SafeTensorInfo> tensors;
    
    // Memory-mapped file data (zero-copy)
    MappedFile mappedFile;
    size_t headerSize = 0;
    
    // Get pointer to start of tensor data section
    const uint8_t* getDataSection() const
    {
        return static_cast<const uint8_t*>(mappedFile.getData()) + 8 + headerSize;
    }
    
public:
    SafeTensorsReader() = default;
    
    // Load from file path
    SlangResult load(const String& path);
    
    // Check if a tensor exists
    bool hasTensor(UnownedStringSlice name) const;
    
    // Get tensor info (shape, dtype, etc.)
    const SafeTensorInfo* getTensorInfo(UnownedStringSlice name) const;
    
    // Get raw pointer to tensor data
    const void* getTensorData(UnownedStringSlice name) const;
    
    // Get tensor count
    Index getTensorCount() const { return tensors.getCount(); }
    
    // List all tensor names
    void getTensorNames(List<String>& outNames) const;
    
    // Read tensor, optionally permute dimensions, and convert to target ElementType.
    // Returns bytes ready to upload to GPU.
    // 
    // permutation: Maps source dimension to destination dimension.
    //   e.g., for [O,I,K,K] -> [I,K,K,O], use {1, 2, 3, 0}
    //   Empty span means no permutation (direct copy).
    SlangResult readTensor(
        UnownedStringSlice name,
        ElementType targetType,
        List<uint8_t>& outData,
        ConstArrayView<int> permutation = {}) const;
};

