#pragma once

#include "core/slang-basic.h"
#include "inference-context.h"

using namespace Slang;

// ============================================================================
// VAE Decoder Test Functions
// ============================================================================

// Test individual ResNet block
SlangResult testVAEResNetBlock(InferencingContext* ctx);

// Test attention block
SlangResult testVAEAttentionBlock(InferencingContext* ctx);

// Test single up block
SlangResult testVAEUpBlock(InferencingContext* ctx);

// Test full VAE decoder (small resolution)
SlangResult testVAEDecoderSmall(InferencingContext* ctx);

// Test full VAE decoder with real SD 1.5 weights
SlangResult testVAEDecoderSD15(InferencingContext* ctx);

