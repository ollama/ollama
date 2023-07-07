package llama

/*
#cgo darwin LDFLAGS: -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
#include <stdlib.h>
#include "llama.h"

struct llama_sample_options
{
	float repeat_penalty;
	float frequency_penalty;
	float presence_penalty;
	float temperature;
	int32_t top_k;
	float top_p;
	float tfs_z;
	float typical_p;
	int mirostat;
	float mirostat_tau;
	float mirostat_eta;
};

llama_token llama_sample(
		struct llama_context *ctx,
		struct llama_token_data *candidates,
		size_t n_candidates,
		const llama_token *last_tokens,
		size_t n_last_tokens,
		struct llama_sample_options *opts)
{
	llama_token_data_array candidates_p = {
		candidates,
		n_candidates,
		false,
	};

	llama_sample_repetition_penalty(
		ctx, &candidates_p,
		last_tokens, n_last_tokens,
		opts->repeat_penalty);

	llama_sample_frequency_and_presence_penalties(
		ctx, &candidates_p,
		last_tokens, n_last_tokens,
		opts->frequency_penalty, opts->presence_penalty);

	if (opts->temperature <= 0) {
		return llama_sample_token_greedy(ctx, &candidates_p);
	}

	if (opts->mirostat == 1) {
		int mirostat_m = 100;
		float mirostat_mu = 2.0f * opts->mirostat_tau;
		llama_sample_temperature(ctx, &candidates_p, opts->temperature);
		return llama_sample_token_mirostat(
			ctx, &candidates_p,
			opts->mirostat_tau, opts->mirostat_eta,
			mirostat_m, &mirostat_mu);
	} else if (opts->mirostat == 2) {
		float mirostat_mu = 2.0f * opts->mirostat_tau;
		llama_sample_temperature(ctx, &candidates_p, opts->temperature);
		return llama_sample_token_mirostat_v2(
			ctx, &candidates_p,
			opts->mirostat_tau, opts->mirostat_eta,
			&mirostat_mu);
	} else {
		llama_sample_top_k(ctx, &candidates_p, opts->top_k, 1);
		llama_sample_tail_free(ctx, &candidates_p, opts->tfs_z, 1);
		llama_sample_typical(ctx, &candidates_p, opts->typical_p, 1);
		llama_sample_top_p(ctx, &candidates_p, opts->top_p, 1);
		llama_sample_temperature(ctx, &candidates_p, opts->temperature);
		return llama_sample_token(ctx, &candidates_p);
	}
}
*/
import "C"
import (
	"errors"
	"io"
	"os"
	"strings"
	"unsafe"

	"github.com/jmorganca/ollama/api"
)

type llama struct {
	params *C.struct_llama_context_params
	model  *C.struct_llama_model
	ctx    *C.struct_llama_context

	api.Options
}

func New(model string, opts api.Options) (*llama, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	llm := llama{Options: opts}

	C.llama_init_backend(C.bool(llm.UseNUMA))

	params := C.llama_context_default_params()
	params.seed = C.uint(llm.Seed)
	params.n_ctx = C.int(llm.NumCtx)
	params.n_batch = C.int(llm.NumBatch)
	params.n_gpu_layers = C.int(llm.NumGPU)
	params.main_gpu = C.int(llm.MainGPU)
	params.low_vram = C.bool(llm.LowVRAM)
	params.f16_kv = C.bool(llm.F16KV)
	params.logits_all = C.bool(llm.LogitsAll)
	params.vocab_only = C.bool(llm.VocabOnly)
	params.use_mmap = C.bool(llm.UseMMap)
	params.use_mlock = C.bool(llm.UseMLock)
	params.embedding = C.bool(llm.EmbeddingOnly)
	llm.params = &params

	cModel := C.CString(model)
	defer C.free(unsafe.Pointer(cModel))

	llm.model = C.llama_load_model_from_file(cModel, params)
	llm.ctx = C.llama_new_context_with_model(llm.model, params)

	// warm up the model
	bos := []C.llama_token{C.llama_token_bos()}
	C.llama_eval(llm.ctx, unsafe.SliceData(bos), C.int(len(bos)), 0, C.int(opts.NumThread))
	C.llama_reset_timings(llm.ctx)

	return &llm, nil
}

func (llm *llama) Close() {
	defer C.llama_free_model(llm.model)
	defer C.llama_free(llm.ctx)

	C.llama_print_timings(llm.ctx)
}

func (llm *llama) Predict(prompt string, fn func(string)) error {
	if tokens := llm.tokenize(prompt); tokens != nil {
		return llm.generate(tokens, fn)
	}

	return errors.New("llama: tokenize")
}

func (llm *llama) tokenize(prompt string) []C.llama_token {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	tokens := make([]C.llama_token, llm.NumCtx)
	if n := C.llama_tokenize(llm.ctx, cPrompt, unsafe.SliceData(tokens), C.int(len(tokens)), true); n > 0 {
		return tokens[:n]
	}

	return nil
}

func (llm *llama) detokenize(tokens ...C.llama_token) string {
	var sb strings.Builder
	for _, token := range tokens {
		sb.WriteString(C.GoString(C.llama_token_to_str(llm.ctx, token)))
	}

	return sb.String()
}

func (llm *llama) generate(tokens []C.llama_token, fn func(string)) error {
	var opts C.struct_llama_sample_options
	opts.repeat_penalty = C.float(llm.RepeatPenalty)
	opts.frequency_penalty = C.float(llm.FrequencyPenalty)
	opts.presence_penalty = C.float(llm.PresencePenalty)
	opts.temperature = C.float(llm.Temperature)
	opts.top_k = C.int(llm.TopK)
	opts.top_p = C.float(llm.TopP)
	opts.tfs_z = C.float(llm.TFSZ)
	opts.typical_p = C.float(llm.TypicalP)
	opts.mirostat = C.int(llm.Mirostat)
	opts.mirostat_tau = C.float(llm.MirostatTau)
	opts.mirostat_eta = C.float(llm.MirostatEta)

	pastTokens := deque[C.llama_token]{capacity: llm.RepeatLastN}

	for C.llama_get_kv_cache_token_count(llm.ctx) < C.int(llm.NumCtx) {
		if retval := C.llama_eval(llm.ctx, unsafe.SliceData(tokens), C.int(len(tokens)), C.llama_get_kv_cache_token_count(llm.ctx), C.int(llm.NumThread)); retval != 0 {
			return errors.New("llama: eval")
		}

		token, err := llm.sample(pastTokens, &opts)
		switch {
		case err != nil:
			return err
		case errors.Is(err, io.EOF):
			return nil
		}

		fn(llm.detokenize(token))

		tokens = []C.llama_token{token}

		pastTokens.PushLeft(token)
	}

	return nil
}

func (llm *llama) sample(pastTokens deque[C.llama_token], opts *C.struct_llama_sample_options) (C.llama_token, error) {
	numVocab := int(C.llama_n_vocab(llm.ctx))
	logits := unsafe.Slice(C.llama_get_logits(llm.ctx), numVocab)

	candidates := make([]C.struct_llama_token_data, 0, numVocab)
	for i := 0; i < numVocab; i++ {
		candidates = append(candidates, C.llama_token_data{
			id:    C.int(i),
			logit: logits[i],
			p:     0,
		})
	}

	token := C.llama_sample(
		llm.ctx,
		unsafe.SliceData(candidates), C.ulong(len(candidates)),
		unsafe.SliceData(pastTokens.Data()), C.ulong(pastTokens.Len()),
		opts)
	if token != C.llama_token_eos() {
		return token, nil
	}

	return 0, io.EOF
}
