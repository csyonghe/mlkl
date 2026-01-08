#pragma once

#include "core/slang-basic.h"
#include "core/slang-string-util.h"

using namespace Slang;

// ============================================================================
// CLIP BPE Tokenizer
// ============================================================================
// 
// Implements the byte-level BPE tokenizer used by CLIP (ViT-L/14).
// This is compatible with OpenAI's CLIP and Stable Diffusion 1.5.
//
// Vocabulary size: 49408
// Special tokens:
//   - <|startoftext|> : 49406
//   - <|endoftext|>   : 49407
// Max sequence length: 77 (including start/end tokens)
//

class CLIPTokenizer
{
public:
    // Special token IDs
    static constexpr int kStartOfText = 49406;
    static constexpr int kEndOfText = 49407;
    static constexpr int kVocabSize = 49408;
    static constexpr int kMaxLength = 77;

private:
    // Vocabulary: token string -> token ID
    Dictionary<String, int> vocab;
    
    // Reverse vocabulary: token ID -> token string
    List<String> idToToken;
    
    // BPE merges: "token1 token2" -> merge rank (lower = higher priority)
    Dictionary<String, int> bpeMerges;
    
    // Byte-to-unicode mapping (for byte-level BPE)
    int byteToUnicode[256];
    Dictionary<int, int> unicodeToByte;
    
    // Cache for BPE results
    mutable Dictionary<String, List<String>> bpeCache;

public:
    CLIPTokenizer() = default;
    
    // Load vocabulary and merges from files
    // vocabPath: path to vocab.json (token -> id mapping)
    // mergesPath: path to merges.txt (BPE merge rules)
    SlangResult load(const String& vocabPath, const String& mergesPath);
    
    // Tokenize a text string
    // Returns token IDs with <|startoftext|> prepended and <|endoftext|> appended
    // Truncates or pads to maxLength
    List<int> encode(const String& text, int maxLength = kMaxLength) const;
    
    // Decode token IDs back to text
    String decode(const List<int>& tokenIds) const;
    
    // Get token ID for a token string
    int getTokenId(const String& token) const;
    
    // Get token string for a token ID
    String getToken(int tokenId) const;
    
    // Check if loaded
    bool isLoaded() const { return vocab.getCount() > 0; }

private:
    // Initialize byte-to-unicode mapping
    void initByteToUnicode();
    
    // Apply BPE to a word
    List<String> bpe(const String& token) const;
    
    // Get the rank of a merge pair (lower = higher priority)
    int getMergeRank(const String& pair) const;
    
    // Pre-tokenize text using CLIP's pattern
    List<String> preTokenize(const String& text) const;
    
    // Convert bytes to BPE-compatible unicode string
    String bytesToBpeString(const String& bytes) const;
    
    // Convert BPE-compatible unicode string back to bytes
    String bpeStringToBytes(const String& bpeStr) const;
    
    // Helper to convert codepoint to UTF-8
    static void appendCodepoint(StringBuilder& sb, int cp);
    
    // Helper to iterate UTF-8 codepoints
    static int nextCodepoint(const char* str, Index& pos, Index len);
};
