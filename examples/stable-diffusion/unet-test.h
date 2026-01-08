#pragma once

#include "inference-context.h"

// Test full SD 1.5 UNet with real weights
SlangResult testSDUNet(InferencingContext* ctx);

// Test individual components
SlangResult testSDResNetBlock(InferencingContext* ctx);
SlangResult testSDSelfAttention(InferencingContext* ctx);
SlangResult testSDCrossAttention(InferencingContext* ctx);
SlangResult testSDSpatialTransformer(InferencingContext* ctx);


