#include "tokenizer-test.h"

#include "example-base/test-utils.h"
#include "tokenizer.h"

#include <cstdio>
#include <initializer_list>

// ============================================================================
// Test CLIP Tokenizer
// ============================================================================

SlangResult testCLIPTokenizer()
{
    printf("Testing CLIP Tokenizer...\n");

    // Load tokenizer
    CLIPTokenizer tokenizer;

    String vocabPath = getTestFilePath("models/vocab.json");
    String mergesPath = getTestFilePath("models/merges.txt");

    SLANG_RETURN_ON_FAIL(tokenizer.load(vocabPath, mergesPath));

    // Test cases with expected token IDs (non-padding tokens only)
    // Padding (49407) is automatically filled to kMaxLength
    // Using const char* with hex escapes for exact byte control (UTF-8)
    struct TestCase
    {
        const char* text;
        std::initializer_list<int> expected;
    };

    // clang-format off
    TestCase testCases[] = {
        // Empty string
        {"", {49406, 49407}},
        
        // Simple words
        {"hello", {49406, 3306, 49407}},
        {"Hello World", {49406, 3306, 1002, 49407}},
        
        // Common prompts
        {"a photo of a cat", {49406, 320, 1125, 539, 320, 2368, 49407}},
        {"a painting of a dog", {49406, 320, 3086, 539, 320, 1929, 49407}},
        {"a beautiful landscape, digital art, trending on artstation",
            {49406, 320, 1215, 5727, 267, 2794, 794, 267, 6087, 525, 1486, 2631, 49407}},
        {"portrait of a woman, highly detailed, 8k, photorealistic",
            {49406, 5352, 539, 320, 2308, 267, 5302, 12609, 267, 279, 330, 267, 1153, 16157, 49407}},
        {"cyberpunk city at night, neon lights, rain",
            {49406, 36896, 1305, 536, 930, 267, 13919, 3073, 267, 2443, 49407}},
        {"fantasy castle on a mountain, epic, cinematic lighting",
            {49406, 5267, 3540, 525, 320, 3965, 267, 4991, 267, 25602, 5799, 49407}},
        
        // Numbers (each digit is separate)
        {"123", {49406, 272, 273, 274, 49407}},
        {"hello123world", {49406, 3306, 272, 273, 274, 1002, 49407}},
        {"2023 new year", {49406, 273, 271, 273, 274, 686, 935, 49407}},
        {"version 2.0", {49406, 3273, 273, 269, 271, 49407}},
        {"50% off", {49406, 276, 271, 260, 1007, 49407}},
        
        // Contractions
        {"don't won't can't", {49406, 847, 713, 1749, 713, 753, 713, 49407}},
        {"it's a test", {49406, 585, 568, 320, 1628, 49407}},
        
        // Punctuation
        {"Hello, World!", {49406, 3306, 267, 1002, 256, 49407}},
        {"What? Why! How...", {49406, 768, 286, 1182, 256, 829, 678, 49407}},
        {"test@email.com", {49406, 1628, 287, 4462, 269, 2464, 49407}},
        
        // Accented characters (Latin Extended) - UTF-8 hex escapes
        // café résumé naïve: é=\xC3\xA9, ï=\xC3\xAF
        {"caf\xC3\xA9 r\xC3\xA9sum\xC3\xA9 na\xC3\xAFve",
            {49406, 15304, 29106, 7054, 4166, 1097, 35689, 563, 49407}},
        
        // CJK characters - UTF-8 hex escapes
        // 日本語: 日=\xE6\x97\xA5 本=\xE6\x9C\xAC 語=\xE8\xAA\x9E
        {"\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E",
            {49406, 37156, 19277, 361, 34002, 508, 49407}},
        
        // Emoji - UTF-8 hex escapes
        // 🎨 = U+1F3A8 = \xF0\x9F\x8E\xA8
        {"emoji \xF0\x9F\x8E\xA8", {49406, 16327, 13461, 49407}},
        
        // Whitespace handling
        {"  hello  world  ", {49406, 3306, 1002, 49407}},
        {"hello\nworld", {49406, 3306, 1002, 49407}},
        {"hello\tworld", {49406, 3306, 1002, 49407}},
    };
    // clang-format on

    int testsPassed = 0;
    int testsFailed = 0;

    for (const auto& tc : testCases)
    {
        auto actual = tokenizer.encode(tc.text);

        // Build expected list padded to kMaxLength
        List<int> expected;
        for (int id : tc.expected)
            expected.add(id);
        while (expected.getCount() < CLIPTokenizer::kMaxLength)
            expected.add(CLIPTokenizer::kEndOfText);

        // Compare
        bool match = (actual.getCount() == expected.getCount());
        if (match)
        {
            for (Index i = 0; i < actual.getCount(); i++)
            {
                if (actual[i] != expected[i])
                {
                    match = false;
                    break;
                }
            }
        }

        if (match)
        {
            testsPassed++;
        }
        else
        {
            testsFailed++;
            if (testsFailed <= 5) // Only show first 5 failures
            {
                printf("  FAIL: '%s'\n", tc.text);
                printf("    Expected: ");
                for (Index i = 0; i < expected.getCount() && i < 10; i++)
                    printf("%d ", expected[i]);
                if (expected.getCount() > 10)
                    printf("...");
                printf("\n");

                printf("    Actual:   ");
                for (Index i = 0; i < actual.getCount() && i < 10; i++)
                    printf("%d ", actual[i]);
                if (actual.getCount() > 10)
                    printf("...");
                printf("\n");
            }
        }
    }

    printf("  Results: %d passed, %d failed\n", testsPassed, testsFailed);

    if (testsFailed > 0)
    {
        printf("testCLIPTokenizer: FAILED\n");
        return SLANG_FAIL;
    }

    printf("testCLIPTokenizer: PASSED\n");
    return SLANG_OK;
}
