package mlxrunner

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/ripple"
	"github.com/ollama/ollama/mlxrunner/sample"
	modeltypes "github.com/ollama/ollama/types/model"
)

type rippleRunnerTensor struct {
	data  []float32
	shape []int
}

func rippleRunnerFixture() (map[string]any, map[string]rippleRunnerTensor) {
	cfg := map[string]any{
		"architectures": []string{"RicursiveForCausalLM"},
		"vocab_size":    19, "hidden_size": 8, "intermediate_size": 12,
		"num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
		"layer_types":    []string{"sliding_attention", "full_attention", "sliding_attention"},
		"sliding_window": 3, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
		"use_valres": true, "use_gated_attention": true, "gate_full_precision": true,
		"ffn_clamp_threshold": 20.0, "embed_scale": 71.55417527999327,
		"max_position_embeddings": 64, "tie_word_embeddings": false,
	}
	shapes := map[string][]int{
		"model.embed_tokens.weight": {19, 8}, "lm_head.weight": {19, 8}, "model.norm.weight": {8},
	}
	for i := range 3 {
		prefix := fmt.Sprintf("model.layers.%d.", i)
		for name, shape := range map[string][]int{
			"input_layernorm.weight": {8}, "post_attention_layernorm.weight": {8},
			"pre_feedforward_layernorm.weight": {8}, "post_feedforward_layernorm.weight": {8},
			"self_attn.q_proj.weight": {16, 8}, "self_attn.k_proj.weight": {8, 8},
			"self_attn.v_proj.weight": {8, 8}, "self_attn.o_proj.weight": {8, 16},
			"self_attn.g_proj.weight": {16, 8}, "self_attn.q_norm.weight": {4},
			"self_attn.k_norm.weight": {4}, "self_attn.value_res_lambda": {1},
			"mlp.gate_proj.weight": {12, 8}, "mlp.up_proj.weight": {12, 8}, "mlp.down_proj.weight": {8, 12},
		} {
			shapes[prefix+name] = shape
		}
	}
	weights := make(map[string]rippleRunnerTensor)
	for name, shape := range shapes {
		size := 1
		for _, dim := range shape {
			size *= dim
		}
		values := make([]float32, size)
		hash := sha256.Sum256([]byte(name))
		phase := int(binary.BigEndian.Uint32(hash[:4]) % 101)
		if strings.HasSuffix(name, "value_res_lambda") {
			layer := strings.Split(name, ".")[2]
			values[0] = map[string]float32{"0": -1.25, "1": 0.75, "2": 1.5}[layer]
		} else {
			for i := range values {
				v := float32((i*37+phase)%101-50) / 50
				switch {
				case strings.Contains(name, "norm.weight"):
					scaled := v * 0.375
					v = 1 + scaled
				case strings.Contains(name, "embed_tokens"):
					v *= 0.045
				case strings.Contains(name, ".mlp.gate_proj") || strings.Contains(name, ".mlp.up_proj"):
					v *= 18
				default:
					v *= 0.3
				}
				values[i] = v
			}
		}
		weights[name] = rippleRunnerTensor{data: values, shape: shape}
	}
	return cfg, weights
}

func writeRippleRunnerModel(t *testing.T) string {
	t.Helper()
	mlxtest.SkipIfUnavailable(t)
	dir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", dir)
	fixtureConfig, fixtureWeights := rippleRunnerFixture()
	config, err := json.Marshal(fixtureConfig)
	if err != nil {
		t.Fatal(err)
	}
	vocab := make(map[string]int)
	for i := range fixtureConfig["vocab_size"].(int) {
		vocab[string(rune('a'+i))] = i
	}
	tokenizer, err := json.Marshal(map[string]any{
		"model": map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{}},
	})
	if err != nil {
		t.Fatal(err)
	}
	name := "ripple-runner-test"
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			weights := make(map[string]*mlx.Array)
			for name, tensor := range fixtureWeights {
				weights[name] = mlx.FromValues(tensor.data, tensor.shape...)
			}
			path := filepath.Join(dir, "model.safetensors")
			if err := mlx.SaveSafetensors(path, weights); err != nil {
				t.Fatal(err)
			}
			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			tensors, err := manifest.NewLayer(f, manifest.MediaTypeImageTensor)
			if err != nil {
				t.Fatal(err)
			}
			tensors.Name = "model.safetensors"
			layers := []manifest.Layer{tensors}
			for name, data := range map[string][]byte{"config.json": config, "tokenizer.json": tokenizer} {
				layer, err := manifest.NewLayer(bytes.NewReader(data), manifest.MediaTypeImageJSON)
				if err != nil {
					t.Fatal(err)
				}
				layer.Name = name
				layers = append(layers, layer)
			}
			meta, err := manifest.NewLayer(strings.NewReader(`{"model_format":"safetensors","model_family":"ripple"}`), manifest.MediaTypeImageConfig)
			if err != nil {
				t.Fatal(err)
			}
			if err := manifest.WriteManifestWithMetadata(modeltypes.ParseName(name), meta, layers, manifest.RunnerMLX, manifest.FormatSafetensors); err != nil {
				t.Fatal(err)
			}
		})
	})
	return name
}

func loadRippleRunner(t *testing.T, name string) *Runner {
	t.Helper()
	r := &Runner{mlxThread: mlxtest.Worker(t)}
	mlxtest.Run(t, func(t *mlxtest.T) {
		if err := r.Load(name); err != nil {
			t.Fatal(err)
		}
		if _, ok := r.Model.(*ripple.Model); !ok {
			t.Fatalf("loaded %T, want *ripple.Model", r.Model)
		}
	})
	t.Cleanup(func() {
		mlxtest.Run(t, func(t *mlxtest.T) {
			r.cache.close()
			r.Close()
		})
	})
	return r
}

func rippleRunnerMemory(t *testing.T) (active int) {
	t.Helper()
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.DefaultStream().Synchronize()
		mlx.ClearCache()
		active = mlx.ActiveMemory()
	})
	return active
}

func rippleRunnerRequest(t *testing.T, r *Runner, prompt string, count int) Request {
	t.Helper()
	request := Request{
		CompletionRequest: CompletionRequest{Prompt: prompt, Options: api.Options{NumPredict: count}},
		Pipeline:          r.TextGenerationPipeline,
		SamplerOpts: sample.Options{
			Temperature: 0, RepeatLastN: 8, RepeatPenalty: 1.1,
			Seed: 1, UseSeed: true, Logprobs: true, TopLogprobs: 3,
		},
		Responses: make(chan CompletionResponse, count+1),
	}
	if err := r.Prepare(&request); err != nil {
		t.Fatal(err)
	}
	if len(request.Tokens) != len(prompt) {
		t.Fatalf("tokenized %q into %d tokens, want %d", prompt, len(request.Tokens), len(prompt))
	}
	return request
}

func runRippleRunnerRequest(t *testing.T, r *Runner, prompt string) (string, []llm.Logprob, int) {
	t.Helper()
	request := rippleRunnerRequest(t, r, prompt, 4)
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	request.Ctx = ctx
	if err := r.runRequest(request); err != nil {
		t.Fatal(err)
	}
	close(request.Responses)
	var text strings.Builder
	var logprobs []llm.Logprob
	var final *CompletionResponse
	for response := range request.Responses {
		if response.Error != nil {
			t.Fatal(response.Error)
		}
		if response.Done {
			if final != nil {
				t.Fatal("duplicate final response")
			}
			final = &response
		}
		text.WriteString(response.Content)
		logprobs = append(logprobs, response.Logprobs...)
	}
	if final == nil || final.EvalCount != 4 || final.DoneReason != 1 || final.PromptEvalCount != len(request.Tokens) || final.PromptEvalCachedCount == nil {
		t.Fatalf("invalid final response: %+v", final)
	}
	if text.Len() != 4 || len(logprobs) != 4 {
		t.Fatalf("empty or incomplete model output: text=%q logprobs=%d", text.String(), len(logprobs))
	}
	for _, lp := range logprobs {
		if math.IsNaN(lp.Logprob) || math.IsInf(lp.Logprob, 0) || lp.Logprob >= 0 || len(lp.TopLogprobs) != 3 {
			t.Fatalf("invalid model log probability: %+v", lp)
		}
		if lp.TopLogprobs[0].Logprob == lp.TopLogprobs[2].Logprob {
			t.Fatalf("model produced uniform logits: %+v", lp)
		}
	}
	return text.String(), logprobs, *final.PromptEvalCachedCount
}

func TestRippleRunnerLoadClose(t *testing.T) {
	name := writeRippleRunnerModel(t)
	warm := loadRippleRunner(t, name)
	mlxtest.Run(t, func(t *mlxtest.T) { warm.Close() })
	baseline := rippleRunnerMemory(t)
	for range 3 {
		r := loadRippleRunner(t, name)
		if loaded := rippleRunnerMemory(t); loaded <= baseline {
			t.Fatalf("load allocated no weights: loaded=%d baseline=%d", loaded, baseline)
		}
		mlxtest.Run(t, func(t *mlxtest.T) { r.Close() })
		if got := rippleRunnerMemory(t); got != baseline {
			t.Fatalf("load/close retained %d bytes above warmed baseline %d", got-baseline, baseline)
		}
	}
}

func TestRippleRunnerRequests(t *testing.T) {
	name := writeRippleRunnerModel(t)
	cold, reused := loadRippleRunner(t, name), loadRippleRunner(t, name)
	const prompt = "ihgfedcba"
	wantText, wantLogprobs, cached := runRippleRunnerRequest(t, cold, prompt)
	if cached != 0 {
		t.Fatalf("fresh runner reused %d tokens", cached)
	}
	runRippleRunnerRequest(t, reused, "abcdefghi")
	gotText, gotLogprobs, cached := runRippleRunnerRequest(t, reused, prompt)
	// These cold forwards have identical chunk schedules; cache-hit forwards do not.
	if cached != 0 || gotText != wantText || !reflect.DeepEqual(gotLogprobs, wantLogprobs) {
		t.Fatalf("independent request inherited state: cached=%d text=%q want=%q", cached, gotText, wantText)
	}
	wantText, wantLogprobs, cached = runRippleRunnerRequest(t, reused, prompt)
	if cached <= 0 || cached >= len(prompt) {
		t.Fatalf("repeat reused %d tokens, want a nonempty prefix with a decode seed", cached)
	}
	baseline := rippleRunnerMemory(t)
	// Settled allocator readings can vary by a few small buffers between requests.
	const memorySlack = 256
	for range 16 {
		gotText, gotLogprobs, gotCached := runRippleRunnerRequest(t, reused, prompt)
		if gotCached != cached || gotText != wantText || !reflect.DeepEqual(gotLogprobs, wantLogprobs) {
			t.Fatal("identical cache-hit schedule changed the response")
		}
		if got := rippleRunnerMemory(t); got > baseline+memorySlack {
			t.Fatalf("repeated request retained %d bytes above warmed baseline %d", got-baseline, baseline)
		}
	}

	request := rippleRunnerRequest(t, reused, prompt, 32)
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	request.Ctx = ctx
	request.Responses = make(chan CompletionResponse)
	responses := make(chan []CompletionResponse, 1)
	go func() {
		var received []CompletionResponse
		for response := range request.Responses {
			received = append(received, response)
			cancel()
		}
		responses <- received
	}()
	err := reused.runRequest(request)
	close(request.Responses)
	received := <-responses
	if !errors.Is(err, context.Canceled) || len(received) == 0 || received[0].Content == "" {
		t.Fatalf("cancel after real output: err=%v responses=%+v", err, received)
	}
	for _, response := range received {
		if response.Done {
			t.Fatal("cancelled generation emitted a successful final response")
		}
	}
	// Retry through the same worker, sampler, and prefix-cache owner.
	runRippleRunnerRequest(t, reused, prompt)
	runRippleRunnerRequest(t, reused, prompt)
	baseline = rippleRunnerMemory(t)
	runRippleRunnerRequest(t, reused, prompt)
	if got := rippleRunnerMemory(t); got > baseline+memorySlack {
		t.Fatalf("post-cancellation request retained %d bytes", got-baseline)
	}
}

func writeRippleDFlash2RunnerModel(t *testing.T) string {
	t.Helper()
	name := writeRippleRunnerModel(t)
	m, err := manifest.ParseNamedManifest(modeltypes.ParseName(name))
	if err != nil {
		t.Fatal(err)
	}
	const draftConfig = `{
		"architectures": ["DFlash2DraftModel"],
		"hidden_size": 8, "num_hidden_layers": 2, "intermediate_size": 16,
		"num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
		"rms_norm_eps": 0.000001, "rope_theta": 10000,
		"vocab_size": 19, "num_target_layers": 3, "block_size": 8,
		"layer_types": ["sliding_attention", "full_attention"],
		"sliding_window": 4, "no_rope_layers": [1], "is_causal": false,
		"dflash_config": {
			"mask_token_id": 18, "target_layer_ids": [0, 2],
			"conv_group_size": 2, "conv_kernel_size": 4,
			"selector_rank": 4, "selector_top_k": 4
		}
	}`
	config, err := manifest.NewLayer(strings.NewReader(draftConfig), manifest.MediaTypeImageJSON)
	if err != nil {
		t.Fatal(err)
	}
	config.Name = "draft/config.json"
	layers := append(m.Layers, config)
	path := filepath.Join(t.TempDir(), "draft.safetensors")
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			weights := make(map[string]*mlx.Array)
			add := func(name string, dims ...int) {
				n, phase := 1, 0
				for _, dim := range dims {
					n *= dim
				}
				for _, c := range name {
					phase += int(c)
				}
				values := make([]float32, n)
				for i := range values {
					values[i] = float32((i*17+phase)%31-15) / 128
					if len(dims) == 1 {
						values[i] = 1
					}
				}
				weights["draft."+name] = mlx.FromValues(values, dims...)
			}
			add("fc.weight", 8, 2*8)
			add("hidden_norm.weight", 8)
			add("norm.weight", 8)
			for i := range 2 {
				prefix := fmt.Sprintf("layers.%d.", i)
				add(prefix+"input_layernorm.weight", 8)
				add(prefix+"post_attention_layernorm.weight", 8)
				add(prefix+"self_attn.q_norm.weight", 4)
				add(prefix+"self_attn.k_norm.weight", 4)
				add(prefix+"self_attn.q_proj.weight", 8, 8)
				add(prefix+"self_attn.k_proj.weight", 4, 8)
				add(prefix+"self_attn.v_proj.weight", 4, 8)
				add(prefix+"self_attn.o_proj.weight", 8, 8)
				add(prefix+"mlp.gate_proj.weight", 16, 8)
				add(prefix+"mlp.up_proj.weight", 16, 8)
				add(prefix+"mlp.down_proj.weight", 8, 16)
				for _, conv := range []string{"attention_conv", "mlp_conv"} {
					add(prefix+conv+".base_kernel", 2, 4, 8)
					add(prefix+conv+".kernel_projection.weight", 2*4*8/2, 8)
				}
			}
			add("candidate_selector.predecessor_codebook", 19, 4)
			add("candidate_selector.successor_codebook", 19, 4)
			add("candidate_selector.hidden_projection.weight", 4, 8)
			if err := mlx.SaveSafetensors(path, weights); err != nil {
				t.Fatal(err)
			}
		})
	})
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	tensors, err := manifest.NewLayer(f, manifest.MediaTypeImageTensor)
	if err != nil {
		t.Fatal(err)
	}
	tensors.Name = "draft/model.safetensors"
	layers = append(layers, tensors)
	metadata, err := json.Marshal(modeltypes.ConfigV2{
		ModelFormat: "safetensors", ModelFamily: "ripple",
		Draft: &modeltypes.Draft{
			ModelFormat: "safetensors", Architecture: "DFlash2DraftModel",
			TensorPrefix: "draft.", Config: "draft/config.json",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	meta, err := manifest.NewLayer(bytes.NewReader(metadata), manifest.MediaTypeImageConfig)
	if err != nil {
		t.Fatal(err)
	}
	name = "ripple-dflash2-runner-test"
	if err := manifest.WriteManifestWithMetadata(modeltypes.ParseName(name), meta, layers, manifest.RunnerMLX, manifest.FormatSafetensors); err != nil {
		t.Fatal(err)
	}
	return name
}

type rippleDraftForward struct {
	offset, contextRows, featureWidth, blockRows int
}

// Observe shapes and calls only; every native operand and selector stays real.
type rippleDFlash2Observer struct {
	model.BlockSelector
	forwards   []rippleDraftForward
	selections int
}

func (d *rippleDFlash2Observer) Forward(b *batch.Batch, targets, drafts []cache.Cache) (*mlx.Array, *mlx.Array) {
	call := rippleDraftForward{offset: int(b.SeqOffsets[0])}
	if b.Hidden != nil {
		call.contextRows, call.featureWidth = b.Hidden.Dim(1), b.Hidden.Dim(2)
	}
	if b.InputIDs != nil {
		call.blockRows = b.InputIDs.Dim(1)
	}
	d.forwards = append(d.forwards, call)
	return d.BlockSelector.Forward(b, targets, drafts)
}

func (d *rippleDFlash2Observer) Select(hidden, logits, anchor *mlx.Array, choose func(ids, scores *mlx.Array) (*mlx.Array, *mlx.Array)) (*mlx.Array, *mlx.Array, *mlx.Array) {
	d.selections++
	return d.BlockSelector.Select(hidden, logits, anchor, choose)
}

func loadRippleDFlash2Runner(t *testing.T, name string) (*Runner, *rippleDFlash2Observer) {
	t.Helper()
	r := loadRippleRunner(t, name)
	if r.spec == nil {
		t.Fatal("combined manifest did not attach a draft")
	}
	draft, ok := r.spec.draft.(model.BlockSelector)
	if !ok {
		t.Fatalf("loaded draft %T, want DFlash2 BlockSelector", r.spec.draft)
	}
	if block, mask := draft.BlockParams(); block != 8 || mask != 18 {
		t.Fatalf("draft block=%d mask=%d, want 8 and 18", block, mask)
	}
	if len(r.spec.targets) != 3 || len(r.spec.draftKV) != 2 || len(r.cache.caches) != 5 {
		t.Fatalf("cache partitions: target=%d draft=%d combined=%d", len(r.spec.targets), len(r.spec.draftKV), len(r.cache.caches))
	}
	if _, ok := r.spec.draftKV[0].(*cache.RotatingKVCache); !ok {
		t.Fatalf("local draft cache is %T", r.spec.draftKV[0])
	}
	if _, ok := r.spec.draftKV[1].(*cache.RotatingKVCache); !ok {
		t.Fatalf("full draft cache is %T", r.spec.draftKV[1])
	}
	observer := &rippleDFlash2Observer{BlockSelector: draft}
	r.spec.draft = observer
	return r, observer
}

func checkRippleDFlash2Caches(t *testing.T, r *Runner, prompt string) {
	t.Helper()
	mlxtest.Run(t, func(t *mlxtest.T) {
		offset := r.spec.targets[0].Offset()
		if offset < len(prompt) {
			t.Fatalf("target stopped before prompt completion: offset=%d prompt=%d", offset, len(prompt))
		}
		for i, c := range r.cache.caches {
			if c.Offset() != offset {
				t.Fatalf("cache %d offset=%d, want settled target offset=%d", i, c.Offset(), offset)
			}
		}
	})
}

type rippleDFlash2Response struct {
	text     string
	cached   int
	forwards []rippleDraftForward
}

func runRippleDFlash2Request(t *testing.T, r *Runner, prompt string, count int) rippleDFlash2Response {
	t.Helper()
	d := r.spec.draft.(*rippleDFlash2Observer)
	d.forwards, d.selections = nil, 0
	request := rippleRunnerRequest(t, r, prompt, count)
	// The shared helper requests logprobs, which would silently park drafting.
	request.SamplerOpts.Logprobs, request.SamplerOpts.TopLogprobs = false, 0
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	request.Ctx = ctx
	if err := r.runRequest(request); err != nil {
		t.Fatal(err)
	}
	close(request.Responses)
	var text strings.Builder
	var final *CompletionResponse
	for response := range request.Responses {
		if response.Error != nil || len(response.Logprobs) != 0 {
			t.Fatalf("unexpected active response: %+v", response)
		}
		text.WriteString(response.Content)
		if response.Done {
			if final != nil {
				t.Fatal("duplicate final response")
			}
			final = &response
		}
	}
	if final == nil || final.EvalCount != count || final.DoneReason != 1 || final.PromptEvalCount != len(request.Tokens) || final.PromptEvalCachedCount == nil || text.Len() != count {
		t.Fatalf("invalid completion: text=%q final=%+v", text.String(), final)
	}
	proposals := 0
	for _, call := range d.forwards {
		if call.contextRows > 0 && call.featureWidth != 16 {
			t.Fatalf("draft did not receive both target taps: %+v", call)
		}
		if call.blockRows > 0 {
			proposals++
			if call.blockRows < 2 || call.blockRows > min(8, count) {
				t.Fatalf("proposal exceeded block/budget: %+v budget=%d", call, count)
			}
		}
	}
	if proposals != d.selections || (count > 1 && proposals == 0) || (count == 1 && proposals != 0) {
		t.Fatalf("active budget=%d: draft forwards=%d selector calls=%d", count, proposals, d.selections)
	}
	checkRippleDFlash2Caches(t, r, prompt)
	return rippleDFlash2Response{text: text.String(), cached: *final.PromptEvalCachedCount, forwards: d.forwards}
}

func TestRippleRunnerDFlash2(t *testing.T) {
	name := writeRippleDFlash2RunnerModel(t)
	const prompt = "abcdefghi"

	// New wiring: normal combined load, target prefill with trailing draft
	// caches, target taps, and a real convolution/selector proposal.
	t.Run("combined_manifest_prefill_proposal", func(t *testing.T) {
		r, d := loadRippleDFlash2Runner(t, name)
		got := runRippleDFlash2Request(t, r, prompt, 9)
		if got.cached != 0 || len(d.forwards) < 2 {
			t.Fatalf("cold request: cached=%d forwards=%+v", got.cached, d.forwards)
		}
		if want := (rippleDraftForward{contextRows: len(prompt) - 1, featureWidth: 16}); d.forwards[0] != want {
			t.Fatalf("prefill handoff=%+v, want %+v", d.forwards[0], want)
		}
		if want := (rippleDraftForward{offset: len(prompt) - 1, blockRows: 8}); d.forwards[1] != want {
			t.Fatalf("first proposal=%+v, want %+v", d.forwards[1], want)
		}
	})

	// Existing runner lifecycle, characterized with the new target/draft pair.
	// Compare identities only after matching cache-hit and proposal schedules.
	t.Run("prefix_branch_reuse", func(t *testing.T) {
		r, _ := loadRippleDFlash2Runner(t, name)
		runRippleDFlash2Request(t, r, prompt, 9)
		runRippleDFlash2Request(t, r, prompt, 9) // warm the repeat schedule
		want := runRippleDFlash2Request(t, r, prompt, 9)
		if want.cached <= 0 || want.cached >= len(prompt) {
			t.Fatalf("repeat cached=%d, want prefix plus an unforwarded seed", want.cached)
		}
		if got := runRippleDFlash2Request(t, r, prompt, 9); !reflect.DeepEqual(got, want) {
			t.Fatalf("matched repeat changed: got=%+v want=%+v", got, want)
		}
		const branch = "abcdejklm"
		if got := runRippleDFlash2Request(t, r, branch, 9); got.cached != 5 {
			t.Fatalf("branch cached=%d, want shared five-token snapshot", got.cached)
		}
		if got := runRippleDFlash2Request(t, r, prompt, 9); !reflect.DeepEqual(got, want) {
			t.Fatalf("returning from branch changed matched schedule: got=%+v want=%+v", got, want)
		}
	})

	t.Run("parked_to_enabled", func(t *testing.T) {
		r, d := loadRippleDFlash2Runner(t, name)
		runRippleRunnerRequest(t, r, prompt)
		if len(d.forwards) == 0 || d.selections != 0 {
			t.Fatalf("parked maintenance: forwards=%+v selections=%d", d.forwards, d.selections)
		}
		for _, call := range d.forwards {
			if call.blockRows != 0 || call.contextRows == 0 || call.featureWidth != 16 {
				t.Fatalf("logprobs must maintain context without proposals: %+v", call)
			}
		}
		checkRippleDFlash2Caches(t, r, prompt)
		if got := runRippleDFlash2Request(t, r, prompt, 9); got.cached <= 0 {
			t.Fatal("enabled request did not reuse parked context")
		}
		runRippleDFlash2Request(t, r, prompt, 9)
		want := runRippleDFlash2Request(t, r, prompt, 9)
		if got := runRippleDFlash2Request(t, r, prompt, 9); !reflect.DeepEqual(got, want) {
			t.Fatalf("enabled repeats changed: got=%+v want=%+v", got, want)
		}
	})

	t.Run("budget_tail", func(t *testing.T) {
		r, d := loadRippleDFlash2Runner(t, name)
		runRippleDFlash2Request(t, r, prompt, 3)
		if len(d.forwards) < 2 || d.forwards[1].blockRows != 3 {
			t.Fatalf("shorter-than-convolution proposal: %+v", d.forwards)
		}
		runRippleDFlash2Request(t, r, prompt, 1)
	})

	t.Run("cancellation_recovery", func(t *testing.T) {
		r, d := loadRippleDFlash2Runner(t, name)
		request := rippleRunnerRequest(t, r, prompt, 16)
		request.SamplerOpts.Logprobs, request.SamplerOpts.TopLogprobs = false, 0
		ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
		defer cancel()
		request.Ctx = ctx
		request.Responses = make(chan CompletionResponse)
		responses := make(chan []CompletionResponse, 1)
		go func() {
			var received []CompletionResponse
			for response := range request.Responses {
				received = append(received, response)
				cancel()
			}
			responses <- received
		}()
		err := r.runRequest(request)
		close(request.Responses)
		received := <-responses
		if !errors.Is(err, context.Canceled) || len(received) == 0 || received[0].Content == "" || d.selections == 0 {
			t.Fatalf("cancel after real speculative output: err=%v responses=%+v selections=%d", err, received, d.selections)
		}
		for _, response := range received {
			if response.Done || response.Error != nil {
				t.Fatalf("cancelled generation emitted terminal response: %+v", response)
			}
		}
		checkRippleDFlash2Caches(t, r, prompt)
		runRippleDFlash2Request(t, r, prompt, 9)
		runRippleDFlash2Request(t, r, prompt, 9)
		want := runRippleDFlash2Request(t, r, prompt, 9)
		if got := runRippleDFlash2Request(t, r, prompt, 9); !reflect.DeepEqual(got, want) {
			t.Fatalf("recovery repeats changed: got=%+v want=%+v", got, want)
		}
	})
}
