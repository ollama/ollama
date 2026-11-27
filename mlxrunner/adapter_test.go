package mlxrunner

import (
	"context"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model/qwen3"
	"github.com/ollama/ollama/mlxrunner/sample"
)

// Each token embeds to e0. The value adapter maps e0 to a different axis
// before it enters KV, so borrowing another adapter's prefix changes logits.
func TestLoRARequestIsolation(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	dir := t.TempDir()
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			const width = 32
			const vocab = 8
			const eps = 1e-6
			tok := newTestTokenizer(t, []int32{7})
			m := &qwen3.Model{Config: &qwen3.Config{HiddenSize: width, HeadDim: width, NumAttentionHeads: 1, NumKeyValueHeads: 1, NumHiddenLayers: 1, IntermediateSize: width, VocabSize: vocab, RMSNormEps: eps, QKNormEps: eps, Scale: float32(1 / math.Sqrt(width)), RopeTheta: 10000}, Layers: make([]*qwen3.Layer, 1)}
			embed, head := make([]float32, vocab*width), make([]float32, vocab*width)
			for i := range vocab {
				embed[i*width] = 1
			}
			head[0], head[width+1], head[2*width+2] = 1, 2, 2
			identity := make([]float32, width*width)
			for i := range width {
				identity[i*width+i] = 1
			}
			norm := mlx.FromValues(slices.Repeat([]float32{1}, width), width)
			w := map[string]*mlx.Array{"model.embed_tokens.weight": mlx.FromValues(embed, vocab, width), "lm_head.weight": mlx.FromValues(head, vocab, width), "model.norm.weight": norm}
			for _, name := range []string{"input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"} {
				w["model.layers.0."+name+".weight"] = norm
			}
			for _, name := range []string{"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"} {
				w["model.layers.0."+name+".weight"] = mlx.Zeros(mlx.DTypeFloat32, width, width)
			}
			w["model.layers.0.self_attn.o_proj.weight"] = mlx.FromValues(identity, width, width)
			if err := m.LoadWeights(w); err != nil {
				t.Fatal(err)
			}
			baseConfig, err := manifest.NewLayer(strings.NewReader(`{"architectures":["Qwen3ForCausalLM"],"hidden_size":32,"num_attention_heads":1,"num_hidden_layers":1,"intermediate_size":32,"vocab_size":8}`), manifest.MediaTypeImageJSON)
			if err != nil {
				t.Fatal(err)
			}
			baseConfig.Name = "config.json"
			r := &Runner{baseManifest: &manifest.Manifest{Layers: []manifest.Layer{baseConfig}}, Model: m, Tokenizer: tok, Sampler: sample.New(4096), contextLength: 4096, parallel: 2}
			defer r.Sampler.Free()
			s := &batchScheduler{r: r, slots: make([]*batchSlot, 2)}
			defer func() {
				if s.snapshotTimer != nil {
					s.snapshotTimer.Stop()
				}
				if s.memory != nil {
					s.memory.Close()
				}
				for _, slot := range s.slots {
					if slot != nil {
						slot.cache.close()
						slot.releaseAdapter()
					}
				}
			}()

			makeAdapter := func(rank, axis int, dtype mlx.DType) manifest.AdapterRef {
				cfg, err := manifest.NewLayer(strings.NewReader(fmt.Sprintf(`{"peft_type":"LORA","r":%d,"lora_alpha":%d,"target_modules":["v_proj"]}`, rank, 2*rank)), manifest.MediaTypeImageJSON)
				if err != nil {
					t.Fatal(err)
				}
				a, b := make([]float32, rank*width), make([]float32, rank*width)
				a[0], b[axis*rank] = 1, 1
				const prefix = "base_model.model.model.layers.0.self_attn.v_proj.lora_"
				path := fmt.Sprintf("%s/%d.safetensors", dir, rank)
				err = mlx.SaveSafetensors(path, map[string]*mlx.Array{prefix + "A.weight": mlx.FromValues(a, rank, width).AsType(dtype), prefix + "B.weight": mlx.FromValues(b, width, rank).AsType(dtype)})
				if err != nil {
					t.Fatal(err)
				}
				weights, err := manifest.NewLayerFromFile(path, manifest.MediaTypePEFTAdapter)
				if err != nil {
					t.Fatal(err)
				}
				return manifest.AdapterRef{Config: cfg.Digest, Weights: weights.Digest}
			}
			a := makeAdapter(2, 1, mlx.DTypeFloat16)
			b := makeAdapter(3, 2, mlx.DTypeBFloat16)
			// Check the full next-token logits against the two RMS norms and
			// W + (alpha/r) B A, independently of the streaming decoder.
			u := 2 / math.Sqrt(1.0/width+eps)
			n := math.Sqrt((1+u*u)/width + eps)
			for i, ref := range []manifest.AdapterRef{a, b} {
				checkpoint, err := r.inspectAdapter(ref)
				if err != nil {
					t.Fatal(err)
				}
				view, err := r.loadAdapter(ref, checkpoint)
				if err != nil {
					t.Fatal(err)
				}
				hidden, _ := view.runner.Model.Forward(&batch.Batch{InputIDs: mlx.FromValues([]int32{0}, 1, 1), SeqOffsets: []int32{0}, SeqQueryLens: []int32{1}}, nil)
				got := view.runner.Model.Unembed(hidden).Floats()
				want := make([]float64, vocab)
				want[0], want[i+1] = 1/n, 2*u/n
				for j, v := range got {
					if math.IsNaN(float64(v)) || math.Abs(float64(v)-want[j]) > 1e-4 {
						t.Fatalf("adapter %d logit %d = %g, want %g", i, j, v, want[j])
					}
				}
				view.weights.Close()
			}
			type trace struct {
				text  string
				probs []llm.Logprob
			}
			run := func(refs ...manifest.AdapterRef) []trace {
				var requests []request
				for _, ref := range refs {
					req := request{Ctx: context.Background(), Tokens: make([]int32, 300), Responses: make(chan CompletionResponse, 16), CompletionRequest: CompletionRequest{Adapter: ref, Options: api.Options{NumPredict: 6}}, SamplerOpts: sample.Options{Temperature: 0, Logprobs: true, TopLogprobs: 3}}
					mlx.Scoped(func() {
						if !s.admit(req) {
							t.Fatal("failed to admit adapter request")
						}
					})
					requests = append(requests, req)
				}
				for i := 0; i < 40 && slices.ContainsFunc(s.slots, func(slot *batchSlot) bool { return slot != nil && slot.busy() }); i++ {
					mlx.Scoped(func() { s.step() })
				}
				var results []trace
				for _, req := range requests {
					var got trace
					var done bool
					for len(req.Responses) > 0 {
						resp := <-req.Responses
						if resp.Error != nil {
							t.Fatal(resp.Error)
						}
						got.text += resp.Content
						got.probs = append(got.probs, resp.Logprobs...)
						done = resp.Done
					}
					if !done || len(got.probs) != 6 || len(got.probs[0].TopLogprobs) != 3 {
						t.Fatalf("incomplete generation: %+v", got)
					}
					results = append(results, got)
				}
				return results
			}
			base := manifest.AdapterRef{}
			isolated := make(map[manifest.AdapterRef]trace)
			for _, tc := range []struct {
				ref  manifest.AdapterRef
				text string
			}{{a, "111111"}, {b, "222222"}, {base, "000000"}} {
				got := run(tc.ref)[0]
				if got.text != tc.text {
					t.Fatalf("got %q, want %q", got.text, tc.text)
				}
				isolated[tc.ref] = got
			}
			for _, refs := range [][]manifest.AdapterRef{{a, b}, {a, a}, {base, a}, {b, base}} {
				for i, got := range run(refs...) {
					if want := isolated[refs[i]]; !reflect.DeepEqual(got, want) {
						t.Fatalf("overlapping request %d for %v differs from isolated result: got %+v, want %+v", i, refs[i], got, want)
					}
				}
			}
		})
	})
}
