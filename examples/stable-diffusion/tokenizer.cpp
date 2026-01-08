#include "tokenizer.h"

#include "core/slang-char-encode.h"
#include "core/slang-io.h"

#include <cctype>
#include <climits>

// ============================================================================
// UTF-8 / UTF-32 Conversion Helpers
// ============================================================================

// Convert UTF-8 string to list of codepoints
static List<Char32> utf8ToCodepoints(const String& str)
{
    List<Char32> codepoints;
    Index pos = 0;
    while (pos < str.getLength())
    {
        Char32 cp = getUnicodePointFromUTF8([&]() -> Byte {
            if (pos < str.getLength())
                return (Byte)str[pos++];
            return 0;
        });
        if (cp == 0 && pos < str.getLength())
            continue; // Skip invalid
        if (cp != 0)
            codepoints.add(cp);
    }
    return codepoints;
}

// Convert codepoints back to UTF-8 string
static String codepointsToUtf8(const List<Char32>& codepoints)
{
    StringBuilder sb;
    for (Index i = 0; i < codepoints.getCount(); i++)
    {
        char buf[6];
        int len = encodeUnicodePointToUTF8(codepoints[i], buf);
        for (int j = 0; j < len; j++)
            sb.appendChar(buf[j]);
    }
    return sb.produceString();
}

// Check if a codepoint is a Unicode letter that can be grouped in words
// (Latin-based scripts with spaces between words)
static bool isUnicodeLetter(Char32 cp)
{
    // ASCII letters
    if (cp >= 'a' && cp <= 'z') return true;
    if (cp >= 'A' && cp <= 'Z') return true;
    
    // Latin Extended (accented letters like é, ñ, ü, etc.)
    if (cp >= 0x00C0 && cp <= 0x00FF) return true;  // Latin-1 Supplement
    if (cp >= 0x0100 && cp <= 0x017F) return true;  // Latin Extended-A
    if (cp >= 0x0180 && cp <= 0x024F) return true;  // Latin Extended-B
    
    // Greek
    if (cp >= 0x0370 && cp <= 0x03FF) return true;
    
    // Cyrillic
    if (cp >= 0x0400 && cp <= 0x04FF) return true;
    
    // NOTE: CJK characters are NOT included here - they are tokenized individually
    
    return false;
}

// Check if a codepoint is CJK (tokenized individually, not grouped)
static bool isUnicodeCJK(Char32 cp)
{
    // CJK (Chinese, Japanese, Korean) - each character is its own token
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;   // CJK Unified Ideographs
    if (cp >= 0x3040 && cp <= 0x309F) return true;   // Hiragana
    if (cp >= 0x30A0 && cp <= 0x30FF) return true;   // Katakana
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;   // Hangul Syllables
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;   // CJK Extension A
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;   // CJK Compatibility Ideographs
    return false;
}

// Check if a codepoint is a digit
static bool isUnicodeDigit(Char32 cp)
{
    return (cp >= '0' && cp <= '9');
}

// Check if a codepoint is whitespace
static bool isUnicodeWhitespace(Char32 cp)
{
    return (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r');
}

// ============================================================================
// UTF-8 Helpers
// ============================================================================

void CLIPTokenizer::appendCodepoint(StringBuilder& sb, int cp)
{
    // Use Slang's encodeUnicodePointToUTF8 for correct encoding
    char buf[6];
    int len = encodeUnicodePointToUTF8(Char32(cp), buf);
    for (int i = 0; i < len; i++)
        sb.appendChar(buf[i]);
}

int CLIPTokenizer::nextCodepoint(const char* str, Index& pos, Index len)
{
    if (pos >= len)
        return -1;

    // Use Slang's getUnicodePointFromUTF8 for correct decoding
    Index startPos = pos;
    Char32 cp = getUnicodePointFromUTF8([&]() -> Byte {
        if (pos < len)
            return (Byte)str[pos++];
        return 0;
    });
    
    // If we didn't advance, there was an error
    if (pos == startPos)
    {
        pos++;
        return -1;
    }
    
    return (int)cp;
}

// ============================================================================
// Byte-to-Unicode Mapping
// ============================================================================

void CLIPTokenizer::initByteToUnicode()
{
    unicodeToByte.clear();

    // Build list of bytes that map to themselves
    List<int> bs;
    for (int i = 33; i <= 126; i++)
        bs.add(i); // '!' to '~'
    for (int i = 161; i <= 172; i++)
        bs.add(i); // '¡' to '¬'
    for (int i = 174; i <= 255; i++)
        bs.add(i); // '®' to 'ÿ'

    List<int> cs;
    for (Index i = 0; i < bs.getCount(); i++)
        cs.add(bs[i]);

    int n = 0;

    // Map remaining bytes to 256+
    for (int b = 0; b < 256; b++)
    {
        bool found = false;
        for (Index i = 0; i < bs.getCount(); i++)
        {
            if (bs[i] == b)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            bs.add(b);
            cs.add(256 + n);
            n++;
        }
    }

    // Build mappings
    for (int b = 0; b < 256; b++)
    {
        byteToUnicode[b] = 0;
    }

    for (Index i = 0; i < bs.getCount(); i++)
    {
        byteToUnicode[bs[i]] = cs[i];
        unicodeToByte[cs[i]] = bs[i];
    }
}

String CLIPTokenizer::bytesToBpeString(const String& bytes) const
{
    StringBuilder sb;
    for (Index i = 0; i < bytes.getLength(); i++)
    {
        unsigned char b = bytes[i];
        int cp = byteToUnicode[b];
        appendCodepoint(sb, cp);
    }
    return sb.produceString();
}

String CLIPTokenizer::bpeStringToBytes(const String& bpeStr) const
{
    StringBuilder sb;
    const char* str = bpeStr.getBuffer();
    Index pos = 0;
    Index len = bpeStr.getLength();

    while (pos < len)
    {
        int cp = nextCodepoint(str, pos, len);
        if (cp < 0)
            continue;

        const int* bytePtr = unicodeToByte.tryGetValue(cp);
        if (bytePtr)
        {
            sb.appendChar((char)*bytePtr);
        }
    }
    return sb.produceString();
}

// ============================================================================
// Loading Vocabulary and Merges
// ============================================================================

SlangResult CLIPTokenizer::load(const String& vocabPath, const String& mergesPath)
{
    initByteToUnicode();

    // Load vocabulary (JSON format)
    {
        String content;
        SLANG_RETURN_ON_FAIL(File::readAllText(vocabPath, content));

        idToToken.setCount(kVocabSize);

        // Simple JSON parsing for {"key": value, ...}
        const char* str = content.getBuffer();
        Index pos = 0;
        Index len = content.getLength();

        // Find opening brace
        while (pos < len && str[pos] != '{')
            pos++;
        pos++;

        while (pos < len)
        {
            // Skip whitespace
            while (pos < len &&
                   (str[pos] == ' ' || str[pos] == '\n' || str[pos] == '\r' || str[pos] == '\t'))
                pos++;

            if (pos >= len || str[pos] == '}')
                break;

            // Parse "key"
            if (str[pos] != '"')
            {
                pos++;
                continue;
            }
            pos++; // Skip opening quote

            StringBuilder keyBuilder;
            while (pos < len && str[pos] != '"')
            {
                if (str[pos] == '\\' && pos + 1 < len)
                {
                    pos++;
                    switch (str[pos])
                    {
                    case 'n':
                        keyBuilder.appendChar('\n');
                        break;
                    case 'r':
                        keyBuilder.appendChar('\r');
                        break;
                    case 't':
                        keyBuilder.appendChar('\t');
                        break;
                    case '\\':
                        keyBuilder.appendChar('\\');
                        break;
                    case '"':
                        keyBuilder.appendChar('"');
                        break;
                    case 'u':
                        // Unicode escape \uXXXX
                        if (pos + 4 < len)
                        {
                            int cp = 0;
                            for (int i = 1; i <= 4; i++)
                            {
                                char c = str[pos + i];
                                int digit = 0;
                                if (c >= '0' && c <= '9')
                                    digit = c - '0';
                                else if (c >= 'a' && c <= 'f')
                                    digit = c - 'a' + 10;
                                else if (c >= 'A' && c <= 'F')
                                    digit = c - 'A' + 10;
                                cp = (cp << 4) | digit;
                            }
                            appendCodepoint(keyBuilder, cp);
                            pos += 4;
                        }
                        break;
                    default:
                        keyBuilder.appendChar(str[pos]);
                        break;
                    }
                }
                else
                {
                    keyBuilder.appendChar(str[pos]);
                }
                pos++;
            }
            pos++; // Skip closing quote

            String key = keyBuilder.produceString();

            // Skip : and whitespace
            while (pos < len && (str[pos] == ':' || str[pos] == ' ' || str[pos] == '\n' ||
                                 str[pos] == '\r' || str[pos] == '\t'))
                pos++;

            // Parse value (integer)
            int value = 0;
            bool negative = false;
            if (pos < len && str[pos] == '-')
            {
                negative = true;
                pos++;
            }
            while (pos < len && str[pos] >= '0' && str[pos] <= '9')
            {
                value = value * 10 + (str[pos] - '0');
                pos++;
            }
            if (negative)
                value = -value;

            vocab[key] = value;
            if (value >= 0 && value < kVocabSize)
            {
                idToToken[value] = key;
            }

            // Skip comma and whitespace
            while (pos < len && (str[pos] == ',' || str[pos] == ' ' || str[pos] == '\n' ||
                                 str[pos] == '\r' || str[pos] == '\t'))
                pos++;
        }
    }

    // Load merges
    {
        String content;
        SLANG_RETURN_ON_FAIL(File::readAllText(mergesPath, content));

        List<UnownedStringSlice> lines;
        StringUtil::calcLines(content.getUnownedSlice(), lines);

        int rank = 0;
        for (Index i = 0; i < lines.getCount(); i++)
        {
            UnownedStringSlice line = lines[i].trim();

            if (line.getLength() == 0)
                continue;

            // Skip header line
            if (line.startsWith("#"))
                continue;

            // Store as "token1 token2" -> rank
            bpeMerges[String(line)] = rank++;
        }
    }

    printf(
        "Loaded tokenizer: %d vocab entries, %d merges\n",
        (int)vocab.getCount(),
        (int)bpeMerges.getCount());
    return SLANG_OK;
}

// ============================================================================
// Pre-tokenization (using UTF-32 codepoints for correctness)
// ============================================================================

List<String> CLIPTokenizer::preTokenize(const String& text) const
{
    List<String> tokens;

    // Convert UTF-8 to codepoints
    List<Char32> codepoints = utf8ToCodepoints(text);
    
    // Convert to lowercase (handles ASCII and some Unicode)
    for (Index i = 0; i < codepoints.getCount(); i++)
    {
        Char32 cp = codepoints[i];
        // ASCII uppercase to lowercase
        if (cp >= 'A' && cp <= 'Z')
            codepoints[i] = cp - 'A' + 'a';
        // Latin-1 Supplement uppercase (À-Þ except × at 0xD7)
        else if (cp >= 0x00C0 && cp <= 0x00DE && cp != 0x00D7)
            codepoints[i] = cp + 0x20;
    }
    
    Index pos = 0;
    Index len = codepoints.getCount();

    while (pos < len)
    {
        // Skip whitespace
        while (pos < len && isUnicodeWhitespace(codepoints[pos]))
            pos++;

        if (pos >= len)
            break;

        Char32 cp = codepoints[pos];
        
        // Check for special tokens (look ahead in UTF-8 for these ASCII sequences)
        // Build a small lookahead string for special token detection
        if (cp == '<')
        {
            // Check for <|startoftext|> or <|endoftext|>
            List<Char32> lookahead;
            for (Index i = pos; i < len && lookahead.getCount() < 16; i++)
                lookahead.add(codepoints[i]);
            String lookaheadStr = codepointsToUtf8(lookahead);
            
            if (lookaheadStr.startsWith("<|startoftext|>"))
            {
                tokens.add("<|startoftext|>");
                pos += 15;
                continue;
            }
            if (lookaheadStr.startsWith("<|endoftext|>"))
            {
                tokens.add("<|endoftext|>");
                pos += 13;
                continue;
            }
        }

        // Check for contractions ('s, 't, 'm, 'd, 're, 've, 'll)
        if (cp == '\'' && pos + 1 < len)
        {
            Char32 next = codepoints[pos + 1];
            if (next == 's' || next == 't' || next == 'm' || next == 'd')
            {
                List<Char32> contraction;
                contraction.add('\'');
                contraction.add(next);
                tokens.add(codepointsToUtf8(contraction));
                pos += 2;
                continue;
            }
            if (pos + 2 < len)
            {
                Char32 next2 = codepoints[pos + 2];
                if ((next == 'r' && next2 == 'e') ||
                    (next == 'v' && next2 == 'e') ||
                    (next == 'l' && next2 == 'l'))
                {
                    List<Char32> contraction;
                    contraction.add('\'');
                    contraction.add(next);
                    contraction.add(next2);
                    tokens.add(codepointsToUtf8(contraction));
                    pos += 3;
                    continue;
                }
            }
        }

        // Collect word (Unicode letters) - CLIP regex: [\p{L}]+
        if (isUnicodeLetter(cp))
        {
            List<Char32> word;
            while (pos < len && isUnicodeLetter(codepoints[pos]))
            {
                word.add(codepoints[pos]);
                pos++;
            }
            tokens.add(codepointsToUtf8(word));
            continue;
        }

        // Collect SINGLE digit - CLIP regex treats each digit separately
        if (isUnicodeDigit(cp))
        {
            List<Char32> digit;
            digit.add(cp);
            tokens.add(codepointsToUtf8(digit));
            pos++;
            continue;
        }

        // Collect SINGLE CJK character - each ideograph is its own token
        if (isUnicodeCJK(cp))
        {
            List<Char32> cjk;
            cjk.add(cp);
            tokens.add(codepointsToUtf8(cjk));
            pos++;
            continue;
        }

        // Collect punctuation/symbols - everything else that's not whitespace/letter/digit/CJK
        // CLIP regex: [^\s\p{L}\p{N}]+
        {
            List<Char32> punct;
            while (pos < len)
            {
                Char32 c = codepoints[pos];
                if (isUnicodeWhitespace(c) || isUnicodeLetter(c) || isUnicodeDigit(c) || isUnicodeCJK(c))
                    break;
                punct.add(c);
                pos++;
            }
            if (punct.getCount() > 0)
            {
                tokens.add(codepointsToUtf8(punct));
            }
        }
    }

    return tokens;
}

// ============================================================================
// BPE Algorithm
// ============================================================================

int CLIPTokenizer::getMergeRank(const String& pair) const
{
    const int* rank = bpeMerges.tryGetValue(pair);
    if (rank)
        return *rank;
    return INT_MAX;
}

List<String> CLIPTokenizer::bpe(const String& token) const
{
    // Check cache
    List<String>* cached = bpeCache.tryGetValue(token);
    if (cached)
        return *cached;

    // Convert token to BPE-compatible unicode and split into characters
    String bpeToken = bytesToBpeString(token);

    if (bpeToken.getLength() == 0)
        return List<String>();

    // Split into individual characters (UTF-8 codepoints)
    List<String> word;
    const char* str = bpeToken.getBuffer();
    Index pos = 0;
    Index len = bpeToken.getLength();

    while (pos < len)
    {
        Index startPos = pos;
        nextCodepoint(str, pos, len);
        word.add(bpeToken.subString(startPos, pos - startPos));
    }

    // CLIP adds "</w>" to the last token to mark end of word
    if (word.getCount() > 0)
    {
        word[word.getCount() - 1] = word[word.getCount() - 1] + "</w>";
    }

    // Iteratively merge most frequent pairs
    while (word.getCount() > 1)
    {
        // Find the pair with lowest rank (highest priority)
        int bestRank = INT_MAX;
        Index bestIdx = 0;

        for (Index i = 0; i < word.getCount() - 1; i++)
        {
            String pair = word[i] + " " + word[i + 1];
            int rank = getMergeRank(pair);
            if (rank < bestRank)
            {
                bestRank = rank;
                bestIdx = i;
            }
        }

        if (bestRank == INT_MAX)
            break;

        // Apply the merge
        String merged = word[bestIdx] + word[bestIdx + 1];
        List<String> newWord;
        for (Index i = 0; i < word.getCount(); i++)
        {
            if (i == bestIdx)
            {
                newWord.add(merged);
                i++; // Skip next token
            }
            else
            {
                newWord.add(word[i]);
            }
        }
        word = newWord;
    }

    // Cache result
    bpeCache[token] = word;

    return word;
}

// ============================================================================
// Encoding / Decoding
// ============================================================================

List<int> CLIPTokenizer::encode(const String& text, int maxLength) const
{
    List<int> tokenIds;

    // Add start of text token
    tokenIds.add(kStartOfText);

    // Pre-tokenize
    List<String> preTokens = preTokenize(text);

    // Apply BPE to each pre-token
    for (Index j = 0; j < preTokens.getCount(); j++)
    {
        List<String> bpeTokens = bpe(preTokens[j]);

        for (Index k = 0; k < bpeTokens.getCount(); k++)
        {
            const int* id = vocab.tryGetValue(bpeTokens[k]);
            if (id)
            {
                tokenIds.add(*id);
            }
            else
            {
                // Try without </w>
                String withoutEnd = bpeTokens[k];
                if (withoutEnd.endsWith("</w>"))
                {
                    withoutEnd = withoutEnd.subString(0, withoutEnd.getLength() - 4);
                }
                id = vocab.tryGetValue(withoutEnd);
                if (id)
                {
                    tokenIds.add(*id);
                }
            }

            // Check length limit
            if (tokenIds.getCount() >= maxLength - 1)
                break;
        }

        if (tokenIds.getCount() >= maxLength - 1)
            break;
    }

    // Add end of text token
    tokenIds.add(kEndOfText);

    // Pad to maxLength
    while (tokenIds.getCount() < maxLength)
    {
        tokenIds.add(kEndOfText);
    }

    return tokenIds;
}

String CLIPTokenizer::decode(const List<int>& tokenIds) const
{
    StringBuilder result;

    for (Index i = 0; i < tokenIds.getCount(); i++)
    {
        int id = tokenIds[i];

        // Skip special tokens
        if (id == kStartOfText || id == kEndOfText)
            continue;

        if (id >= 0 && id < idToToken.getCount())
        {
            String token = idToToken[id];

            // Remove </w> suffix and add space
            if (token.endsWith("</w>"))
            {
                token = token.subString(0, token.getLength() - 4);
                result.append(bpeStringToBytes(token));
                result.appendChar(' ');
            }
            else
            {
                result.append(bpeStringToBytes(token));
            }
        }
    }

    // Trim trailing space
    String str = result.produceString();
    while (str.getLength() > 0 && str[str.getLength() - 1] == ' ')
    {
        str = str.subString(0, str.getLength() - 1);
    }

    return str;
}

int CLIPTokenizer::getTokenId(const String& token) const
{
    const int* id = vocab.tryGetValue(token);
    if (id)
        return *id;
    return -1;
}

String CLIPTokenizer::getToken(int tokenId) const
{
    if (tokenId >= 0 && tokenId < idToToken.getCount())
        return idToToken[tokenId];
    return String();
}
