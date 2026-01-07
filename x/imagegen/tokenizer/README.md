# Tokenizer

Fast, correct tokenizer for LLM inference supporting BPE, SentencePiece, and WordPiece algorithms.

## Features

- **BPE (Byte Pair Encoding)** - GPT-2/Llama style with byte-level encoding
- **SentencePiece** - Gemma style with `▁` space handling
- **WordPiece** - BERT style with `##` continuation tokens
- **Parallel encoding** - Automatic parallelization for inputs >4KB
- **HuggingFace compatible** - Loads `tokenizer.json` directly

## Usage

```go
import "github.com/ollama/ollama/x/imagegen/tokenizer"

// Load from HuggingFace model directory
tok, err := tokenizer.Load("./weights/Llama-3.2-1B")
if err != nil {
    log.Fatal(err)
}

// Encode text to token IDs
ids := tok.Encode("Hello, world!", false) // false = don't add BOS

// Decode back to text
text := tok.Decode(ids)

// Check special tokens
if tok.IsEOS(ids[len(ids)-1]) {
    // End of sequence
}
```

## Performance

Benchmarks on Apple M3 Max:

| Input Size | Encode | Decode | Tokens |
|------------|--------|--------|--------|
| 1 KB | 14.5 MB/s | 267 MB/s | 231 |
| 10 KB | 10.9 MB/s | 321 MB/s | 2,301 |
| 100 KB | 8.9 MB/s | 311 MB/s | 23,001 |
| 1 MB | 9.6 MB/s | 321 MB/s | 230,001 |

Comparison with other implementations (10 MB input):

| Implementation | Encode Speed | Notes |
|----------------|--------------|-------|
| Engine (this) | ~10 MB/s | stdlib RE2, parallel >4KB |
| tiktoken (Rust) | ~17 MB/s | Highly optimized regex |
| llama.cpp (C++) | ~2 MB/s | Single-threaded only |
| Ollama (Go) | ~2-3 MB/s | regexp2 backtracking |

## Correctness

The tokenizer matches HuggingFace transformers exactly. Verified with:

- 82 rigorous test cases for Gemma (SentencePiece)
- 458 fuzz test cases covering Unicode edge cases
- Full 0x00-0xFF byte roundtrip for BPE

Run tests:
```bash
go test ./tokenizer/... -v
```

## Architecture

```
Load(path)
    │
    ├─ tokenizer.json → loadFromTokenizerJSON()
    │                      ├─ Parse vocab, merges, added_tokens
    │                      ├─ Detect type: BPE / SentencePiece / WordPiece
    │                      ├─ Compile pretokenizer regex (BPE only)
    │                      └─ Load special tokens from config files
    │
    └─ vocab.json + merges.txt → loadVocabMerges()

Encode(text)
    │
    ├─ Split by special tokens
    ├─ Apply pretokenizer regex (BPE) or space→▁ (SentencePiece)
    ├─ For each chunk:
    │      ├─ Fast path: single token lookup
    │      └─ Slow path: BPE merge algorithm
    └─ Parallel for inputs >4KB

Decode(ids)
    │
    ├─ Look up each token
    └─ Apply inverse transform:
           ├─ BPE: byte-level decode (0x0100 → 0x00, etc.)
           ├─ SentencePiece: ▁→space, <0xNN>→byte
           └─ WordPiece: strip ## prefix
```

## Key Implementation Details

### BPE Byte-Level Encoding

GPT-2 style encoding maps bytes to Unicode codepoints to handle arbitrary binary data:

```go
// Precomputed table: byte → rune
var byteToRune [256]rune // 0x00→0x0100, 0x20→0x0120, etc.
```

### Pretokenizer Regex

HuggingFace patterns use PCRE features not supported by Go's RE2. We rewrite:

```go
// PCRE (HuggingFace)
`\s+(?!\S)|\s+`

// RE2 (Go) - with post-processing for whitespace boundaries
`\s+`
```

### Special Token Handling

Special tokens are matched greedily (longest first) before pretokenization:

```go
// Sorted by length, checked with HasPrefix
for _, tok := range sortedSpecialTokens {
    if strings.HasPrefix(remaining, tok) {
        // Found special token
    }
}
```

## Performance Opportunities

Potential optimizations not yet implemented:

| Optimization | Expected Gain | Complexity |
|--------------|---------------|------------|
| Aho-Corasick for special tokens | 2-3x for many special tokens | Medium |
| Custom regex engine (like tiktoken) | 1.5-2x | High |
| SIMD byte scanning | 1.3-1.5x for pretokenizer | Medium |
| Assembly BPE merge loop | 1.2-1.5x | High |
| Memoization for repeated substrings | Variable | Low |

Current bottleneck is the pretokenizer regex (~60% of encode time). tiktoken achieves ~17 MB/s with a hand-tuned Rust regex engine.

## Not Yet Implemented

| Feature | Used By | Notes |
|---------|---------|-------|
| Unigram tokenizer | T5, ALBERT, mBART | Different algorithm (not BPE) |
| Unicode normalizers | Some multilingual models | NFD, NFKC, lowercase, etc. |
| Custom pretokenizers | Model-specific | Beyond standard patterns |

Most HuggingFace models use BPE or SentencePiece, which are fully supported. WordPiece (BERT-style) is also supported with standard `[UNK]` fallback for out-of-vocabulary characters.

## Files

| File | Description |
|------|-------------|
| `tokenizer.go` | Main implementation (~1000 lines) |
| `tokenizer_test.go` | Tests and benchmarks |
| `testdata/` | Mini tokenizer for unit tests |
