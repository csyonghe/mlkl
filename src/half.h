#pragma once

#include <cstdint>
#include <cstring>

// ============================================================================
// Half-Precision Float (FP16) Conversion Utilities
// ============================================================================
// IEEE 754 half-precision binary16 format:
// - 1 sign bit, 5 exponent bits, 10 mantissa bits
// - Range: ~6.1e-5 to 65504 (with subnormals down to ~5.96e-8)
// ============================================================================

// Convert single-precision float (F32) to half-precision float (F16)
inline uint16_t floatToHalf(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFF;

    if (exponent <= 0)
    {
        // Denormalized or zero
        if (exponent < -10)
            return (uint16_t)sign;  // Too small, flush to zero
        mantissa = (mantissa | 0x800000) >> (1 - exponent);
        return (uint16_t)(sign | (mantissa >> 13));
    }
    else if (exponent == 0xFF - 127 + 15)
    {
        // Inf or NaN
        if (mantissa == 0)
            return (uint16_t)(sign | 0x7C00);  // Inf
        else
            return (uint16_t)(sign | 0x7C00 | (mantissa >> 13));  // NaN
    }
    else if (exponent > 30)
    {
        // Overflow to Inf
        return (uint16_t)(sign | 0x7C00);
    }
    
    return (uint16_t)(sign | (exponent << 10) | (mantissa >> 13));
}

// Convert half-precision float (F16) to single-precision float (F32)
inline float halfToFloat(uint16_t h)
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
            float f;
            memcpy(&f, &result, sizeof(float));
            return f;
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
            exponent = exponent + (127 - 15);
            uint32_t result = sign | (exponent << 23) | (mantissa << 13);
            float f;
            memcpy(&f, &result, sizeof(float));
            return f;
        }
    }
    else if (exponent == 31)
    {
        // Inf or NaN
        uint32_t result = sign | 0x7F800000 | (mantissa << 13);
        float f;
        memcpy(&f, &result, sizeof(float));
        return f;
    }
    else
    {
        // Normalized
        exponent = exponent + (127 - 15);
        uint32_t result = sign | (exponent << 23) | (mantissa << 13);
        float f;
        memcpy(&f, &result, sizeof(float));
        return f;
    }
}

// ============================================================================
// BFloat16 Conversion Utilities
// ============================================================================
// BFloat16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
// Simply truncates/extends the upper 16 bits of a float32
// ============================================================================

// Convert single-precision float (F32) to BFloat16
inline uint16_t floatToBFloat16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

// Convert BFloat16 to single-precision float (F32)
inline float bfloat16ToFloat(uint16_t bf16)
{
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}


