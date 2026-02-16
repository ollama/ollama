# Tokenizer

Tokenizer for LLM inference supporting BPE, SentencePiece, and WordPiece algorithms. The goal of this package is to see if a pure Go tokenizer can be fast and correct. It primarily supports the `imagegen` models however it (or parts of it) could be considered to replace Ollama's tokenizer in the `model` package.

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
| Ollama (Go) | ~2-3 MB/s | regexp2 backtracking |

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
