#pragma once

#include "core/slang-basic.h"
#include "inference-context.h"
#include "torch-reader.h"

using namespace Slang;

enum class ActivationFunction
{
    None,
    ReLU,
    SiLU
};

const char* getActivationFuncName(ActivationFunction func);
