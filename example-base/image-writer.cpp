#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "example-base.h"
#include "stb_image_write.h"

#include <cstdint>

void writeImagePNG(const char* filename, int width, int height, int numChannels, const void* data)
{
    int strideInBytes = width * numChannels * sizeof(uint8_t);
    stbi_write_png(filename, width, height, numChannels, data, strideInBytes);
}

void writeImagePNG(
    const char* filename,
    int width,
    int height,
    int channels,
    const Slang::List<float>& imageData)
{
    Slang::List<uint8_t> outputImageData8Bit;
    outputImageData8Bit.setCount(width * height * channels);
    for (int i = 0; i < imageData.getCount(); i++)
    {
        float v = imageData[i];
        v = (Slang::Math::Clamp(v, -1.0f, 1.0f) + 1.0) * 0.5f;
        outputImageData8Bit[i] = static_cast<uint8_t>(v * 255.0f);
    }
    writeImagePNG(filename, width, height, channels, outputImageData8Bit.getBuffer());
}
