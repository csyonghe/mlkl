#include "json.h"

#include "core/slang-diagnostic-sink.h"

using namespace Slang;

RefPtr<ParsedJsonContent> parseJson(UnownedStringSlice jsonText)
{
    RefPtr<ParsedJsonContent> result = new ParsedJsonContent();
    auto sourceManager = &result->sourceManager;

    SourceFile* sourceFile =
        sourceManager->createSourceFileWithString(PathInfo::makeUnknown(), jsonText);
    SourceView* sourceView = sourceManager->createSourceView(sourceFile, nullptr, SourceLoc());

    DiagnosticSink sink;
    JSONLexer lexer;
    lexer.init(sourceView, &sink);
    result->container = new JSONContainer(sourceManager);
    JSONBuilder builder(result->container);
    JSONParser parser;
    if (SLANG_FAILED(parser.parse(&lexer, sourceView, &builder, &sink)))
        return nullptr;
    result->rootValue = builder.getRootValue();
    return result;
}
