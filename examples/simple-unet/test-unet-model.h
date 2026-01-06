#pragma once

#include "inference-context.h"

// Test the first up block (up0) against PyTorch reference.
// Requires running dump_unet_reference.py first to generate ref_*.bin files.
SlangResult testUpBlock0(InferencingContext* ctx);

// Test the first down block (down0) against PyTorch reference.
// Requires running dump_unet_reference.py first to generate ref_*.bin files.
SlangResult testDownBlock0(InferencingContext* ctx);

// Test just the initial conv (conv0) against PyTorch reference.
// Requires running dump_unet_reference.py first to generate ref_*.bin files.
SlangResult testInitialConv(InferencingContext* ctx);

// Test just the time embedding kernel against PyTorch reference.
// Requires running dump_unet_reference.py first to generate ref_*.bin files.
SlangResult testTimeEmbedding(InferencingContext* ctx);

// Test UNet model by comparing C++ output against PyTorch reference.
// Requires running dump_unet_reference.py first to generate ref_*.bin files.
SlangResult testUNetModelAgainstPyTorch(InferencingContext* ctx);

