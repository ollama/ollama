#ifndef OGA_H
#define OGA_H

#include "oga_types.h"
#include "dynamic.h"

// Function pointer declarations for ORT GenAI C API
// Each function gets: (1) extern function pointer, (2) static inline wrapper

// --- Config ---
extern OgaResult* (*OgaCreateConfig_)(const char* config_path, OgaConfig** out);
static inline OgaResult* OgaCreateConfig(const char* config_path, OgaConfig** out) { return OgaCreateConfig_(config_path, out); }

extern void (*OgaDestroyConfig_)(OgaConfig* config);
static inline void OgaDestroyConfig(OgaConfig* config) { OgaDestroyConfig_(config); }

extern OgaResult* (*OgaConfigClearProviders_)(OgaConfig* config);
static inline OgaResult* OgaConfigClearProviders(OgaConfig* config) { return OgaConfigClearProviders_(config); }

extern OgaResult* (*OgaConfigAppendProvider_)(OgaConfig* config, const char* provider_name);
static inline OgaResult* OgaConfigAppendProvider(OgaConfig* config, const char* provider_name) { return OgaConfigAppendProvider_(config, provider_name); }

extern OgaResult* (*OgaConfigSetProviderOption_)(OgaConfig* config, const char* provider_name, const char* key, const char* value);
static inline OgaResult* OgaConfigSetProviderOption(OgaConfig* config, const char* provider_name, const char* key, const char* value) { return OgaConfigSetProviderOption_(config, provider_name, key, value); }

// --- Model ---
extern OgaResult* (*OgaCreateModelFromConfig_)(const OgaConfig* config, OgaModel** out);
static inline OgaResult* OgaCreateModelFromConfig(const OgaConfig* config, OgaModel** out) { return OgaCreateModelFromConfig_(config, out); }

extern void (*OgaDestroyModel_)(OgaModel* model);
static inline void OgaDestroyModel(OgaModel* model) { OgaDestroyModel_(model); }

// --- Tokenizer ---
extern OgaResult* (*OgaCreateTokenizer_)(const OgaModel* model, OgaTokenizer** out);
static inline OgaResult* OgaCreateTokenizer(const OgaModel* model, OgaTokenizer** out) { return OgaCreateTokenizer_(model, out); }

extern void (*OgaDestroyTokenizer_)(OgaTokenizer* tokenizer);
static inline void OgaDestroyTokenizer(OgaTokenizer* tokenizer) { OgaDestroyTokenizer_(tokenizer); }

extern OgaResult* (*OgaTokenizerEncode_)(const OgaTokenizer* tokenizer, const char* str, OgaSequences* sequences);
static inline OgaResult* OgaTokenizerEncode(const OgaTokenizer* tokenizer, const char* str, OgaSequences* sequences) { return OgaTokenizerEncode_(tokenizer, str, sequences); }

extern OgaResult* (*OgaTokenizerDecode_)(const OgaTokenizer* tokenizer, const int32_t* tokens, size_t token_count, const char** out_string);
static inline OgaResult* OgaTokenizerDecode(const OgaTokenizer* tokenizer, const int32_t* tokens, size_t token_count, const char** out_string) { return OgaTokenizerDecode_(tokenizer, tokens, token_count, out_string); }

// --- Sequences ---
extern OgaResult* (*OgaCreateSequences_)(OgaSequences** out);
static inline OgaResult* OgaCreateSequences(OgaSequences** out) { return OgaCreateSequences_(out); }

extern void (*OgaDestroySequences_)(OgaSequences* sequences);
static inline void OgaDestroySequences(OgaSequences* sequences) { OgaDestroySequences_(sequences); }

extern size_t (*OgaSequencesCount_)(const OgaSequences* sequences);
static inline size_t OgaSequencesCount(const OgaSequences* sequences) { return OgaSequencesCount_(sequences); }

extern size_t (*OgaSequencesGetSequenceCount_)(const OgaSequences* sequences, size_t index);
static inline size_t OgaSequencesGetSequenceCount(const OgaSequences* sequences, size_t index) { return OgaSequencesGetSequenceCount_(sequences, index); }

extern const int32_t* (*OgaSequencesGetSequenceData_)(const OgaSequences* sequences, size_t index);
static inline const int32_t* OgaSequencesGetSequenceData(const OgaSequences* sequences, size_t index) { return OgaSequencesGetSequenceData_(sequences, index); }

// --- Tokenizer Stream ---
extern OgaResult* (*OgaCreateTokenizerStream_)(const OgaTokenizer* tokenizer, OgaTokenizerStream** out);
static inline OgaResult* OgaCreateTokenizerStream(const OgaTokenizer* tokenizer, OgaTokenizerStream** out) { return OgaCreateTokenizerStream_(tokenizer, out); }

extern void (*OgaDestroyTokenizerStream_)(OgaTokenizerStream* stream);
static inline void OgaDestroyTokenizerStream(OgaTokenizerStream* stream) { OgaDestroyTokenizerStream_(stream); }

extern OgaResult* (*OgaTokenizerStreamDecode_)(OgaTokenizerStream* stream, int32_t token, const char** out);
static inline OgaResult* OgaTokenizerStreamDecode(OgaTokenizerStream* stream, int32_t token, const char** out) { return OgaTokenizerStreamDecode_(stream, token, out); }

// --- Generator Params ---
extern OgaResult* (*OgaCreateGeneratorParams_)(const OgaModel* model, OgaGeneratorParams** out);
static inline OgaResult* OgaCreateGeneratorParams(const OgaModel* model, OgaGeneratorParams** out) { return OgaCreateGeneratorParams_(model, out); }

extern void (*OgaDestroyGeneratorParams_)(OgaGeneratorParams* params);
static inline void OgaDestroyGeneratorParams(OgaGeneratorParams* params) { OgaDestroyGeneratorParams_(params); }

extern OgaResult* (*OgaGeneratorParamsSetSearchNumber_)(OgaGeneratorParams* params, const char* name, double value);
static inline OgaResult* OgaGeneratorParamsSetSearchNumber(OgaGeneratorParams* params, const char* name, double value) { return OgaGeneratorParamsSetSearchNumber_(params, name, value); }

extern OgaResult* (*OgaGeneratorParamsSetSearchBool_)(OgaGeneratorParams* params, const char* name, bool value);
static inline OgaResult* OgaGeneratorParamsSetSearchBool(OgaGeneratorParams* params, const char* name, bool value) { return OgaGeneratorParamsSetSearchBool_(params, name, value); }

// --- Generator ---
extern OgaResult* (*OgaCreateGenerator_)(const OgaModel* model, const OgaGeneratorParams* params, OgaGenerator** out);
static inline OgaResult* OgaCreateGenerator(const OgaModel* model, const OgaGeneratorParams* params, OgaGenerator** out) { return OgaCreateGenerator_(model, params, out); }

extern void (*OgaDestroyGenerator_)(OgaGenerator* generator);
static inline void OgaDestroyGenerator(OgaGenerator* generator) { OgaDestroyGenerator_(generator); }

extern OgaResult* (*OgaGenerator_AppendTokenSequences_)(OgaGenerator* generator, const OgaSequences* sequences);
static inline OgaResult* OgaGenerator_AppendTokenSequences(OgaGenerator* generator, const OgaSequences* sequences) { return OgaGenerator_AppendTokenSequences_(generator, sequences); }

extern OgaResult* (*OgaGenerator_GenerateNextToken_)(OgaGenerator* generator);
static inline OgaResult* OgaGenerator_GenerateNextToken(OgaGenerator* generator) { return OgaGenerator_GenerateNextToken_(generator); }

extern bool (*OgaGenerator_IsDone_)(OgaGenerator* generator);
static inline bool OgaGenerator_IsDone(OgaGenerator* generator) { return OgaGenerator_IsDone_(generator); }

extern OgaResult* (*OgaGenerator_GetNextTokens_)(const OgaGenerator* generator, const int32_t** out, size_t* out_count);
static inline OgaResult* OgaGenerator_GetNextTokens(const OgaGenerator* generator, const int32_t** out, size_t* out_count) { return OgaGenerator_GetNextTokens_(generator, out, out_count); }

// --- Error handling ---
extern const char* (*OgaResultGetError_)(const OgaResult* result);
static inline const char* OgaResultGetError(const OgaResult* result) { return OgaResultGetError_(result); }

extern void (*OgaDestroyResult_)(OgaResult* result);
static inline void OgaDestroyResult(OgaResult* result) { OgaDestroyResult_(result); }

extern void (*OgaDestroyString_)(const char* str);
static inline void OgaDestroyString(const char* str) { OgaDestroyString_(str); }

extern void (*OgaShutdown_)(void);
static inline void OgaShutdown(void) { OgaShutdown_(); }

// Symbol loader — called after dlopen/LoadLibrary
int oga_dynamic_load_symbols(oga_dynamic_handle handle);

#endif // OGA_H
