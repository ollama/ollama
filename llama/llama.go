package llama

/*
#cgo CPPFLAGS: -O3 -DNDEBUG=1
#cgo CXXFLAGS: -std=c++11
#cgo darwin CPPFLAGS: -DGGML_USE_METAL=1 -DGGML_METAL_NDEBUG=1
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
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"
	"unicode/utf8"
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

	C.llama_backend_init(C.bool(llm.UseNUMA))

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
	if llm.model == nil {
		return nil, errors.New("failed to load model")
	}

	llm.ctx = C.llama_new_context_with_model(llm.model, params)
	if llm.ctx == nil {
		return nil, errors.New("failed to create context")
	}

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

func (llm *llama) Predict(ctx []int, prompt string, fn func(api.GenerateResponse)) error {
	if input := llm.tokenize(prompt); input != nil {
		embd := make([]C.llama_token, len(ctx))
		for i := range ctx {
			embd[i] = C.llama_token(ctx[i])
		}

		return llm.generate(append(embd, input...), fn)
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

func (llm *llama) generate(input []C.llama_token, fn func(api.GenerateResponse)) error {
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

	output := deque[C.llama_token]{capacity: llm.NumCtx}

	context := deque[int]{capacity: llm.NumCtx / 2}
	for _, in := range input {
		context.PushLeft(int(in))
	}

	var b bytes.Buffer
	for C.llama_get_kv_cache_token_count(llm.ctx) < C.int(llm.NumCtx) {
		if retval := C.llama_eval(llm.ctx, unsafe.SliceData(input), C.int(len(input)), C.llama_get_kv_cache_token_count(llm.ctx), C.int(llm.NumThread)); retval != 0 {
			return errors.New("llama: eval")
		}

		token, err := llm.sample(output, &opts)
		if errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return err
		}

		b.WriteString(llm.detokenize(token))
		if utf8.Valid(b.Bytes()) || b.Len() >= utf8.UTFMax {
			// call the callback
			fn(api.GenerateResponse{
				Response: b.String(),
			})

			output.PushLeft(token)
			context.PushLeft(int(token))
			b.Reset()
		}

		input = []C.llama_token{token}
	}

	dur := func(ms float64) time.Duration {
		d, err := time.ParseDuration(fmt.Sprintf("%fms", ms))
		if err != nil {
			panic(err)
		}

		return d
	}

	timings := C.llama_get_timings(llm.ctx)
	fn(api.GenerateResponse{
		Done:               true,
		Context:            context.Data(),
		PromptEvalCount:    int(timings.n_p_eval),
		PromptEvalDuration: dur(float64(timings.t_p_eval_ms)),
		EvalCount:          int(timings.n_eval),
		EvalDuration:       dur(float64(timings.t_eval_ms)),
	})

	return nil
}

func (llm *llama) sample(output deque[C.llama_token], opts *C.struct_llama_sample_options) (C.llama_token, error) {
	numVocab := int(C.llama_n_vocab(llm.ctx))
	logits := unsafe.Slice(C.llama_get_logits(llm.ctx), numVocab)

	candidates := deque[C.struct_llama_token_data]{capacity: numVocab}
	for i := 0; i < candidates.Cap(); i++ {
		candidates.PushLeft(C.struct_llama_token_data{
			id:    C.int(i),
			logit: logits[i],
			p:     0,
		})
	}

	token := C.llama_sample(
		llm.ctx,
		unsafe.SliceData(candidates.Data()), C.size_t(candidates.Len()),
		unsafe.SliceData(output.Data()), C.size_t(output.Len()),
		opts)
	if token != C.llama_token_eos() {
		return token, nil
	}

	return 0, io.EOF
}
