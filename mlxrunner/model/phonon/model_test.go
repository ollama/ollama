package phonon

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestConfig(t *testing.T) {
	const checkpoint = `{"flow_lm":{"d_model":512,"num_heads":8,"num_layers":6,"max_period":10000,"flow_depth":6,"gelu":"erf","time_rms_norm":"rms"},"mimi":{"sample_rate":24000,"frame_rate":12.5,"dilation_base":2,"transformer_num_heads":8,"transformer_num_layers":2,"transformer_context":250,"transformer_max_period":10000,"gelu":"erf"},"conditioners":[{"type":"lut","name":"num_speakers","lut":{"possible_values":["0","1","2"]}},{"type":"continuous","name":"duration_delta","continuous":{"dim":128}}],"fuser":{"sum":["num_speakers","duration_delta"],"prepend":["transcript_in_segment","speaker_wavs"]},"temp":0.8,"lsd_decode_steps":1,"eos_threshold":4.0}`
	var cfg config
	if err := json.Unmarshal([]byte(checkpoint), &cfg); err != nil {
		t.Fatal(err)
	}
	if err := cfg.validate(); err != nil {
		t.Fatal(err)
	}
	if cfg.EOSThreshold != 4 || cfg.Temperature != .8 {
		t.Fatalf("lost checkpoint sampling settings: %+v", cfg)
	}
	for _, tc := range []struct {
		name   string
		change func(*config)
	}{
		{"GELU", func(c *config) { c.FlowLM.GELU = "relu" }},
		{"time norm", func(c *config) { c.FlowLM.TimeRMSNorm = "other" }},
		{"sample rate", func(c *config) { c.Mimi.SampleRate = 48000 }},
		{"layers", func(c *config) { c.FlowLM.NumLayers = 0 }},
		{"multiple LSD steps", func(c *config) { c.DecodeSteps = 2 }},
		{"negative temperature", func(c *config) { c.Temperature = -1 }},
		{"NaN threshold", func(c *config) { c.EOSThreshold = float32(math.NaN()) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := cfg
			tc.change(&c)
			if c.validate() == nil {
				t.Fatal("accepted unsupported configuration")
			}
		})
	}
}

func TestPrereleaseConfigRejected(t *testing.T) {
	var cfg config
	if err := json.Unmarshal([]byte(`{"flow_lm":{"num_heads":12,"num_layers":6,"max_period":10000,"flow_depth":6},"mimi":{"sample_rate":24000,"frame_rate":12.5}}`), &cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.validate() == nil {
		t.Fatal("accepted prerelease Phonon configuration")
	}
}

func TestTimeNorm(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		values := []float32{1, 2, 3, 6}
		x, alpha := mlx.FromValues(values, 4), mlx.FromValues([]float32{2, 2, 2, 2}, 4)
		for _, tc := range []struct {
			norm        string
			denominator float64
		}{
			{"rms", math.Sqrt(12.5 + 1e-5)},
		} {
			out := timeNorm(x, alpha, tc.norm)
			mlx.Eval(out)
			for i, got := range out.Floats() {
				want := 2 * float64(values[i]) / tc.denominator
				if math.Abs(float64(got)-want) > 1e-5 {
					t.Errorf("%s[%d] = %v, want %v", tc.norm, i, got, want)
				}
			}
		}
	})
}
