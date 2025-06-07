package ollamarunner

import (
	"testing"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/runner/common"
)

// testBackend implements ml.Backend with minimal functionality required for tests.
type testBackend struct{}

func (b *testBackend) Config() fs.Config             { return testConfig{} }
func (b *testBackend) Get(string) ml.Tensor          { return nil }
func (b *testBackend) NewContext() ml.Context        { return &testContext{} }
func (b *testBackend) NewContextSize(int) ml.Context { return &testContext{} }

// testConfig is a stub implementation of fs.Config used by testBackend.
type testConfig struct{}

func (testConfig) Architecture() string                  { return "" }
func (testConfig) String(string, ...string) string       { return "" }
func (testConfig) Uint(string, ...uint32) uint32         { return 0 }
func (testConfig) Float(string, ...float32) float32      { return 0 }
func (testConfig) Bool(string, ...bool) bool             { return false }
func (testConfig) Strings(string, ...[]string) []string  { return nil }
func (testConfig) Ints(string, ...[]int32) []int32       { return nil }
func (testConfig) Floats(string, ...[]float32) []float32 { return nil }

type testContext struct{}

func (c *testContext) Empty(dtype ml.DType, shape ...int) ml.Tensor {
	sz := 1
	for _, s := range shape {
		sz *= s
	}
	return &testTensor{dtype: dtype, data: make([]float32, sz), shape: shape}
}
func (c *testContext) Zeros(dtype ml.DType, shape ...int) ml.Tensor { return c.Empty(dtype, shape...) }
func (c *testContext) FromFloatSlice(s []float32, shape ...int) (ml.Tensor, error) {
	t := c.Empty(ml.DTypeF32, shape...).(*testTensor)
	copy(t.data, s)
	return t, nil
}

func (c *testContext) FromIntSlice(s []int32, shape ...int) (ml.Tensor, error) {
	f := make([]float32, len(s))
	for i, v := range s {
		f[i] = float32(v)
	}
	out, _ := c.FromFloatSlice(f, shape...)
	out.(*testTensor).dtype = ml.DTypeI32
	return out, nil
}

func (c *testContext) Arange(start, stop, step float32, dtype ml.DType) ml.Tensor {
	return c.Empty(dtype, int((stop-start)/step))
}
func (c *testContext) Forward(...ml.Tensor) ml.Context { return c }
func (c *testContext) Compute(...ml.Tensor)            {}
func (c *testContext) Reserve() error                  { return nil }
func (c *testContext) MaxGraphNodes() int              { return 0 }
func (c *testContext) Close()                          {}
func (c *testContext) Input() ml.Context               { return c }
func (c *testContext) Layer(int) ml.Context            { return c }

type testTensor struct {
	ml.Tensor
	dtype ml.DType
	data  []float32
	shape []int
}

func (t *testTensor) Dim(n int) int    { return t.shape[n] }
func (t *testTensor) Stride(n int) int { return 0 }
func (t *testTensor) Shape() []int     { return t.shape }
func (t *testTensor) DType() ml.DType  { return t.dtype }
func (t *testTensor) Bytes() []byte    { return nil }
func (t *testTensor) Floats() []float32 {
	out := make([]float32, len(t.data))
	copy(out, t.data)
	return out
}
func (t *testTensor) Neg(ctx ml.Context) ml.Tensor { return nil }
func (t *testTensor) Add(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	out, _ := ctx.(*testContext).FromFloatSlice(nil, len(t.data))
	return out
}
func (t *testTensor) Mul(ctx ml.Context, t2 ml.Tensor) ml.Tensor            { return nil }
func (t *testTensor) Mulmat(ctx ml.Context, t2 ml.Tensor) ml.Tensor         { return nil }
func (t *testTensor) MulmatFullPrec(ctx ml.Context, t2 ml.Tensor) ml.Tensor { return nil }
func (t *testTensor) MulmatID(ctx ml.Context, t2, ids ml.Tensor) ml.Tensor  { return nil }
func (t *testTensor) Softmax(ctx ml.Context) ml.Tensor                      { return nil }
func (t *testTensor) LayerNorm(ctx ml.Context, w, b ml.Tensor, e float32) ml.Tensor {
	return nil
}

func (t *testTensor) View(ctx ml.Context, offset int, shape ...int) ml.Tensor {
	return ctx.(*testContext).Empty(t.dtype, shape...)
}

func (t *testTensor) Copy(ctx ml.Context, dest ml.Tensor) ml.Tensor {
	copy(dest.(*testTensor).data, t.data)
	return nil
}

// testModel implements model.Model and model.TextProcessor.
type testModel struct {
	model.Base
	decode  map[int32]string
	logits  [][]float32
	call    int
	backend ml.Backend
}

func (f *testModel) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	idx := f.call
	if idx >= len(f.logits) {
		idx = len(f.logits) - 1
	}
	f.call++
	return ctx.FromFloatSlice(f.logits[idx], len(f.logits[idx]))
}

func (f *testModel) Backend() ml.Backend {
	if f.backend == nil {
		f.backend = &testBackend{}
	}
	return f.backend
}

func (f *testModel) Encode(string, bool) ([]int32, error) { return nil, nil }
func (f *testModel) Decode(ids []int32) (string, error) {
	var s string
	for _, id := range ids {
		s += f.decode[id]
	}
	return s, nil
}
func (f *testModel) Is(id int32, sp model.Special) bool { return false }
func (f *testModel) Vocabulary() *model.Vocabulary      { return &model.Vocabulary{} }

var (
	_ model.Model         = (*testModel)(nil)
	_ model.TextProcessor = (*testModel)(nil)
)

func TestFlushPending(t *testing.T) {
	tests := []struct {
		name              string
		pendingResponses  []llm.CompletionResponse
		expectedResponses []llm.CompletionResponse
		description       string
	}{
		{
			name: "ongoing_response",
			pendingResponses: []llm.CompletionResponse{
				{Content: "Hello", Done: false},
				{Content: " world", Done: false},
				{Content: "!", Done: false},
			},
			expectedResponses: []llm.CompletionResponse{
				{Content: "Hello", Done: false},
				{Content: " world", Done: false},
				{Content: "!", Done: false},
			},
			description: "All responses should be flushed when the last one has Done=true",
		},
		{
			name: "complete_response_with_done",
			pendingResponses: []llm.CompletionResponse{
				{Content: "Hello", Done: false},
				{Content: " world", Done: false},
				{Content: "!", Done: true},
			},
			expectedResponses: []llm.CompletionResponse{
				{Content: "Hello", Done: false},
				{Content: " world", Done: false},
				{Content: "!", Done: true},
			},
			description: "All responses should be flushed when the last one has Done=true",
		},
		{
			name: "incomplete_unicode_at_end_with_done",
			pendingResponses: []llm.CompletionResponse{
				{Content: "Hello", Done: false},
				{Content: string([]byte{0xF0, 0x9F}), Done: true}, // Incomplete emoji with Done=true
			},
			expectedResponses: []llm.CompletionResponse{
				{Content: "Hello", Done: false},
				{Content: "", Done: true}, // Content is trimmed but response is still sent with Done=true
			},
			description: "When Done=true, incomplete Unicode at the end should be trimmed but response with Done flag sent",
		},
		{
			name: "split_unicode_across_responses",
			pendingResponses: []llm.CompletionResponse{
				{Content: "Hello " + string([]byte{0xF0, 0x9F}), Done: false}, // First part of 😀
				{Content: string([]byte{0x98, 0x80}) + " world!", Done: true}, // Second part of 😀 and more text
			},
			expectedResponses: []llm.CompletionResponse{
				{Content: "Hello ", Done: false},  // Incomplete Unicode trimmed
				{Content: "😀 world!", Done: true}, // Complete emoji in second response
			},
			description: "Unicode split across responses should be handled correctly when Done=true",
		},
		{
			name: "empty_final_response_with_done",
			pendingResponses: []llm.CompletionResponse{
				{Content: "Complete response", Done: false},
				{Content: "", Done: true}, // Empty response with Done=true
			},
			expectedResponses: []llm.CompletionResponse{
				{Content: "Complete response", Done: false},
				{Content: "", Done: true}, // Should still be sent because Done=true
			},
			description: "Empty final response with Done=true should still be sent",
		},
		{
			name: "done_reason_preserved",
			pendingResponses: []llm.CompletionResponse{
				{Content: "Response", Done: false},
				{Content: " complete", Done: true, DoneReason: llm.DoneReasonStop}, // With specific DoneReason
			},
			expectedResponses: []llm.CompletionResponse{
				{Content: "Response", Done: false},
				{Content: " complete", Done: true, DoneReason: llm.DoneReasonStop}, // DoneReason should be preserved
			},
			description: "DoneReason should be preserved in the final response",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a test sequence with a buffered channel to collect responses
			seq := &Sequence{
				pendingResponses: tt.pendingResponses,
				responses:        make(chan llm.CompletionResponse, len(tt.expectedResponses)),
				quit:             make(chan bool, 1),
			}

			result := common.FlushPending(seq)

			// Verify that flushPending returned true (success)
			if !result {
				t.Fatal("flushPending returned false, expected true")
			}

			// Close the responses channel to allow range iteration
			close(seq.responses)

			// Collect the responses
			var actualResponses []llm.CompletionResponse
			for resp := range seq.responses {
				actualResponses = append(actualResponses, resp)
			}

			// Verify the number of responses
			if len(actualResponses) != len(tt.expectedResponses) {
				t.Fatalf("%s: got %d responses, want %d responses",
					tt.description, len(actualResponses), len(tt.expectedResponses))
			}

			// Verify each response matches the expected one
			for i, expected := range tt.expectedResponses {
				if i >= len(actualResponses) {
					t.Fatalf("%s: missing response at index %d", tt.description, i)
					continue
				}

				actual := actualResponses[i]

				// Verify content
				if actual.Content != expected.Content {
					t.Errorf("%s: response[%d].Content = %q, want %q",
						tt.description, i, actual.Content, expected.Content)
				}

				// Verify Done flag
				if actual.Done != expected.Done {
					t.Errorf("%s: response[%d].Done = %v, want %v",
						tt.description, i, actual.Done, expected.Done)
				}

				// Verify DoneReason if specified
				if actual.DoneReason != expected.DoneReason {
					t.Errorf("%s: response[%d].DoneReason = %v, want %v",
						tt.description, i, actual.DoneReason, expected.DoneReason)
				}
			}

			// Verify pendingResponses was cleared
			if len(seq.pendingResponses) != 0 {
				t.Errorf("%s: pendingResponses not cleared, has %d items",
					tt.description, len(seq.pendingResponses))
			}
		})
	}
}
