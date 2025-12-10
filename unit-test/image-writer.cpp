#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdint>

void writeImagePNG(
    const char* filename,
    int width,
    int height,
    int numChannels,
    const void* data)
{
    int strideInBytes = width * numChannels * sizeof(uint8_t);
    stbi_write_png(filename, width, height, numChannels, data, strideInBytes);
}