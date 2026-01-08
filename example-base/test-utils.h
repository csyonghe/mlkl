#pragma once

#include "core/slang-basic.h"
#include "core/slang-io.h"

using namespace Slang;

// ============================================================================
// Test Utilities
// ============================================================================
// Common utilities for testing across all examples.

// Search for a test file in common locations (current dir, parent dirs, etc.)
// Returns empty string if not found.
String getTestFilePath(String subpath);

// Load binary tensor data (float32) from file
SlangResult loadBinaryTensor(const String& path, List<float>& outData);

// Check if two float arrays are approximately equal
// rtol: relative tolerance
// atol: absolute tolerance
// Returns true if all elements satisfy: |actual - expected| <= atol + rtol * |expected|
bool checkApproxEqual(
    const List<float>& actual,
    const List<float>& expected,
    float rtol = 1e-3f,
    float atol = 1e-5f);


