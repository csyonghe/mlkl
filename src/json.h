#pragma once

#include "core/slang-basic.h"
#include "core/slang-json-value.h"
#include "core/slang-source-loc.h"

struct ParsedJsonContent : public Slang::RefObject
{
    Slang::RefPtr<Slang::JSONContainer> container;
    Slang::JSONValue rootValue;
    Slang::SourceManager sourceManager;
};

Slang::RefPtr<ParsedJsonContent> parseJson(Slang::UnownedStringSlice jsonText);