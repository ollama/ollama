#include "oga.h"

// Function pointer globals — initialized to NULL, resolved by oga_dynamic_load_symbols

// Config
OgaResult* (*OgaCreateConfig_)(const char* config_path, OgaConfig** out) = NULL;
void (*OgaDestroyConfig_)(OgaConfig* config) = NULL;
OgaResult* (*OgaConfigClearProviders_)(OgaConfig* config) = NULL;
OgaResult* (*OgaConfigAppendProvider_)(OgaConfig* config, const char* provider_name) = NULL;
OgaResult* (*OgaConfigSetProviderOption_)(OgaConfig* config, const char* provider_name, const char* key, const char* value) = NULL;

// Model
OgaResult* (*OgaCreateModelFromConfig_)(const OgaConfig* config, OgaModel** out) = NULL;
void (*OgaDestroyModel_)(OgaModel* model) = NULL;

// Tokenizer
OgaResult* (*OgaCreateTokenizer_)(const OgaModel* model, OgaTokenizer** out) = NULL;
void (*OgaDestroyTokenizer_)(OgaTokenizer* tokenizer) = NULL;
OgaResult* (*OgaTokenizerEncode_)(const OgaTokenizer* tokenizer, const char* str, OgaSequences* sequences) = NULL;
OgaResult* (*OgaTokenizerDecode_)(const OgaTokenizer* tokenizer, const int32_t* tokens, size_t token_count, const char** out_string) = NULL;

// Sequences
OgaResult* (*OgaCreateSequences_)(OgaSequences** out) = NULL;
void (*OgaDestroySequences_)(OgaSequences* sequences) = NULL;
size_t (*OgaSequencesCount_)(const OgaSequences* sequences) = NULL;
size_t (*OgaSequencesGetSequenceCount_)(const OgaSequences* sequences, size_t index) = NULL;
const int32_t* (*OgaSequencesGetSequenceData_)(const OgaSequences* sequences, size_t index) = NULL;

// Tokenizer Stream
OgaResult* (*OgaCreateTokenizerStream_)(const OgaTokenizer* tokenizer, OgaTokenizerStream** out) = NULL;
void (*OgaDestroyTokenizerStream_)(OgaTokenizerStream* stream) = NULL;
OgaResult* (*OgaTokenizerStreamDecode_)(OgaTokenizerStream* stream, int32_t token, const char** out) = NULL;

// Generator Params
OgaResult* (*OgaCreateGeneratorParams_)(const OgaModel* model, OgaGeneratorParams** out) = NULL;
void (*OgaDestroyGeneratorParams_)(OgaGeneratorParams* params) = NULL;
OgaResult* (*OgaGeneratorParamsSetSearchNumber_)(OgaGeneratorParams* params, const char* name, double value) = NULL;
OgaResult* (*OgaGeneratorParamsSetSearchBool_)(OgaGeneratorParams* params, const char* name, bool value) = NULL;

// Generator
OgaResult* (*OgaCreateGenerator_)(const OgaModel* model, const OgaGeneratorParams* params, OgaGenerator** out) = NULL;
void (*OgaDestroyGenerator_)(OgaGenerator* generator) = NULL;
OgaResult* (*OgaGenerator_AppendTokenSequences_)(OgaGenerator* generator, const OgaSequences* sequences) = NULL;
OgaResult* (*OgaGenerator_GenerateNextToken_)(OgaGenerator* generator) = NULL;
bool (*OgaGenerator_IsDone_)(OgaGenerator* generator) = NULL;
OgaResult* (*OgaGenerator_GetNextTokens_)(const OgaGenerator* generator, const int32_t** out, size_t* out_count) = NULL;

// Error handling
const char* (*OgaResultGetError_)(const OgaResult* result) = NULL;
void (*OgaDestroyResult_)(OgaResult* result) = NULL;
void (*OgaDestroyString_)(const char* str) = NULL;
void (*OgaShutdown_)(void) = NULL;

int oga_dynamic_load_symbols(oga_dynamic_handle handle) {
    // Config
    CHECK_LOAD(handle, OgaCreateConfig);
    CHECK_LOAD(handle, OgaDestroyConfig);
    CHECK_LOAD(handle, OgaConfigClearProviders);
    CHECK_LOAD(handle, OgaConfigAppendProvider);
    CHECK_LOAD(handle, OgaConfigSetProviderOption);

    // Model
    CHECK_LOAD(handle, OgaCreateModelFromConfig);
    CHECK_LOAD(handle, OgaDestroyModel);

    // Tokenizer
    CHECK_LOAD(handle, OgaCreateTokenizer);
    CHECK_LOAD(handle, OgaDestroyTokenizer);
    CHECK_LOAD(handle, OgaTokenizerEncode);
    CHECK_LOAD(handle, OgaTokenizerDecode);

    // Sequences
    CHECK_LOAD(handle, OgaCreateSequences);
    CHECK_LOAD(handle, OgaDestroySequences);
    CHECK_LOAD(handle, OgaSequencesCount);
    CHECK_LOAD(handle, OgaSequencesGetSequenceCount);
    CHECK_LOAD(handle, OgaSequencesGetSequenceData);

    // Tokenizer Stream
    CHECK_LOAD(handle, OgaCreateTokenizerStream);
    CHECK_LOAD(handle, OgaDestroyTokenizerStream);
    CHECK_LOAD(handle, OgaTokenizerStreamDecode);

    // Generator Params
    CHECK_LOAD(handle, OgaCreateGeneratorParams);
    CHECK_LOAD(handle, OgaDestroyGeneratorParams);
    CHECK_LOAD(handle, OgaGeneratorParamsSetSearchNumber);
    CHECK_LOAD(handle, OgaGeneratorParamsSetSearchBool);

    // Generator
    CHECK_LOAD(handle, OgaCreateGenerator);
    CHECK_LOAD(handle, OgaDestroyGenerator);
    CHECK_LOAD(handle, OgaGenerator_AppendTokenSequences);
    CHECK_LOAD(handle, OgaGenerator_GenerateNextToken);
    CHECK_LOAD(handle, OgaGenerator_IsDone);
    CHECK_LOAD(handle, OgaGenerator_GetNextTokens);

    // Error handling
    CHECK_LOAD(handle, OgaResultGetError);
    CHECK_LOAD(handle, OgaDestroyResult);
    CHECK_LOAD(handle, OgaDestroyString);
    CHECK_LOAD(handle, OgaShutdown);

    return 0;
}
