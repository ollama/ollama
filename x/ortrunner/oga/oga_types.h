#ifndef OGA_TYPES_H
#define OGA_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Opaque types matching onnxruntime-genai C API
typedef struct OgaResult OgaResult;
typedef struct OgaModel OgaModel;
typedef struct OgaConfig OgaConfig;
typedef struct OgaGenerator OgaGenerator;
typedef struct OgaGeneratorParams OgaGeneratorParams;
typedef struct OgaTokenizer OgaTokenizer;
typedef struct OgaTokenizerStream OgaTokenizerStream;
typedef struct OgaSequences OgaSequences;

#endif // OGA_TYPES_H
