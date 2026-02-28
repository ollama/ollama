//go:build mlx

package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/qwen3_5"
	"github.com/ollama/ollama/x/tokenizer"
)

type stubModel struct{}

func (stubModel) Forward(*mlx.Array, []cache.Cache) *mlx.Array { return nil }
func (stubModel) Unembed(*mlx.Array) *mlx.Array                { return nil }
func (stubModel) NumLayers() int                               { return 0 }
func (stubModel) Tokenizer() *tokenizer.Tokenizer              { return nil }
func (stubModel) LoadWeights(map[string]*mlx.Array) error      { return nil }

func TestResolveSamplingConfigDefaults(t *testing.T) {
	trueValue := true
	falseValue := false

	tests := []struct {
		name  string
		model base.Model
		req   Request
		want  samplingConfig
	}{
		{
			name:  "generic model uses api defaults",
			model: stubModel{},
			req:   Request{},
			want: samplingConfig{
				temperature:      0.8,
				topP:             0.9,
				minP:             0.0,
				topK:             40,
				repeatLastN:      64,
				repeatPenalty:    1.1,
				presencePenalty:  0.0,
				frequencyPenalty: 0.0,
			},
		},
		{
			name:  "qwen3.5 defaults to thinking profile when think unset",
			model: &qwen3_5.Model{},
			req:   Request{},
			want: samplingConfig{
				temperature:      1.0,
				topP:             0.95,
				minP:             0.0,
				topK:             20,
				repeatLastN:      64,
				repeatPenalty:    1.0,
				presencePenalty:  1.5,
				frequencyPenalty: 0.0,
			},
		},
		{
			name:  "qwen3.5 thinking disabled defaults",
			model: &qwen3_5.Model{},
			req:   Request{TextCompletionsRequest: TextCompletionsRequest{Think: &falseValue}},
			want: samplingConfig{
				temperature:      0.7,
				topP:             0.8,
				minP:             0.0,
				topK:             20,
				repeatLastN:      64,
				repeatPenalty:    1.0,
				presencePenalty:  1.5,
				frequencyPenalty: 0.0,
			},
		},
		{
			name:  "qwen3.5 thinking enabled defaults",
			model: &qwen3_5.Model{},
			req:   Request{TextCompletionsRequest: TextCompletionsRequest{Think: &trueValue}},
			want: samplingConfig{
				temperature:      1.0,
				topP:             0.95,
				minP:             0.0,
				topK:             20,
				repeatLastN:      64,
				repeatPenalty:    1.0,
				presencePenalty:  1.5,
				frequencyPenalty: 0.0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveSamplingConfig(tt.model, tt.req); got != tt.want {
				t.Fatalf("resolveSamplingConfig() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestResolveSamplingConfigOverridesSpecifiedValues(t *testing.T) {
	trueValue := true
	temperature := float32(0.4)
	topP := float32(0.6)
	minP := float32(0.05)
	topK := 12
	repeatLastN := 32
	repeatPenalty := float32(1.1)
	presencePenalty := float32(0.7)
	frequencyPenalty := float32(0.2)

	got := resolveSamplingConfig(stubModel{}, Request{
		TextCompletionsRequest: TextCompletionsRequest{
			Think: &trueValue,
			Options: struct {
				Temperature      *float32 `json:"temperature"`
				TopP             *float32 `json:"top_p"`
				MinP             *float32 `json:"min_p"`
				TopK             *int     `json:"top_k"`
				RepeatLastN      *int     `json:"repeat_last_n"`
				RepeatPenalty    *float32 `json:"repeat_penalty"`
				PresencePenalty  *float32 `json:"presence_penalty"`
				FrequencyPenalty *float32 `json:"frequency_penalty"`
				MaxTokens        int      `json:"max_tokens"`
				NumPredict       int      `json:"num_predict"`
			}{
				Temperature:      &temperature,
				TopP:             &topP,
				MinP:             &minP,
				TopK:             &topK,
				RepeatLastN:      &repeatLastN,
				RepeatPenalty:    &repeatPenalty,
				PresencePenalty:  &presencePenalty,
				FrequencyPenalty: &frequencyPenalty,
			},
		},
	})

	want := samplingConfig{
		temperature:      temperature,
		topP:             topP,
		minP:             minP,
		topK:             topK,
		repeatLastN:      repeatLastN,
		repeatPenalty:    repeatPenalty,
		presencePenalty:  presencePenalty,
		frequencyPenalty: frequencyPenalty,
	}
	if got != want {
		t.Fatalf("resolveSamplingConfig() = %+v, want %+v", got, want)
	}
}

func TestResolveSamplingConfigMatchesGenericDefaults(t *testing.T) {
	want := api.DefaultOptions()
	got := defaultSamplingConfig(stubModel{}, nil)

	if got.temperature != want.Temperature ||
		got.topP != want.TopP ||
		got.minP != want.MinP ||
		got.topK != want.TopK ||
		got.repeatLastN != want.RepeatLastN ||
		got.repeatPenalty != want.RepeatPenalty ||
		got.presencePenalty != want.PresencePenalty ||
		got.frequencyPenalty != want.FrequencyPenalty {
		t.Fatalf("defaultSamplingConfig() = %+v, want api defaults %+v", got, want)
	}
}
