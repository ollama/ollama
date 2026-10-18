package phonon

import (
	"encoding/json"
	"math"
	"testing"
)

func TestConfig(t *testing.T) {
	const checkpoint = `{"flow_lm":{"num_heads":12,"num_layers":6,"max_period":10000,"flow_depth":6},"mimi":{"sample_rate":24000,"frame_rate":12.5,"dilation_base":2,"transformer_num_heads":8,"transformer_num_layers":2,"transformer_context":250,"transformer_max_period":10000},"temp":0.8,"lsd_decode_steps":1,"eos_threshold":4.0}`
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
