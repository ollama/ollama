package model

import (
	"reflect"
	"testing"
)

func TestParseHFGenerationDefaults(t *testing.T) {
	defaults, err := ParseHFGenerationDefaults([]byte(`{
		"top_k": 40.0,
		"top_p": 0.7,
		"min_p": 0,
		"temperature": 0.6,
		"repetition_penalty": 1.05,
		"penalty_repeat": 1.4,
		"presence_penalty": 0.1,
		"frequency_penalty": 0.2,
		"penalty_last_n": 64.0
	}`))
	if err != nil {
		t.Fatal(err)
	}

	check := func(key string, want any) {
		t.Helper()
		if got := defaults[key]; got != want {
			t.Fatalf("%s = %#v, want %#v", key, got, want)
		}
	}

	check("top_k", int64(40))
	check("top_p", float64(0.7))
	check("min_p", float64(0))
	check("temperature", float64(0.6))
	check("repeat_penalty", float64(1.05))
	check("presence_penalty", float64(0.1))
	check("frequency_penalty", float64(0.2))
	check("repeat_last_n", int64(64))
}

func TestParseHFGenerationDefaultsIgnoresUnsupportedValues(t *testing.T) {
	defaults, err := ParseHFGenerationDefaults([]byte(`{
		"top_p": 0.8,
		"do_sample": true,
		"eos_token_id": 128001,
		"pad_token_id": 128002,
		"max_new_tokens": 2048,
		"mirostat_tau": 5.0,
		"typical_p": 0.95
	}`))
	if err != nil {
		t.Fatal(err)
	}

	if got := defaults["top_p"]; got != float64(0.8) {
		t.Fatalf("top_p = %#v, want %#v", got, float64(0.8))
	}

	for _, key := range []string{"do_sample", "eos_token_id", "pad_token_id", "max_new_tokens", "mirostat_tau", "typical_p"} {
		if _, ok := defaults[key]; ok {
			t.Fatalf("%s should be ignored", key)
		}
	}
}

func TestParseHFGenerationDefaultsSkipsInvalidValues(t *testing.T) {
	defaults, err := ParseHFGenerationDefaults([]byte(`{
		"top_k": "40",
		"top_p": 0.8
	}`))
	if err != nil {
		t.Fatal(err)
	}

	if _, ok := defaults["top_k"]; ok {
		t.Fatal("top_k should be skipped when it is not numeric")
	}
	if got := defaults["top_p"]; got != float64(0.8) {
		t.Fatalf("top_p = %#v, want %#v", got, float64(0.8))
	}
}

func TestParseHFGenerationDefaultsNullValues(t *testing.T) {
	tests := []struct {
		name string
		data string
		want GenerationDefaults
	}{
		{
			name: "all null",
			data: `{
				"top_k": null, "top_p": null, "min_p": null,
				"temperature": null, "repetition_penalty": null,
				"repeat_penalty": null, "penalty_repeat": null,
				"presence_penalty": null, "frequency_penalty": null,
				"repeat_last_n": null, "penalty_last_n": null
			}`,
		},
		{
			name: "null alongside numeric value",
			data: `{"temperature": null, "top_p": null, "top_k": null, "min_p": 0.05}`,
			want: GenerationDefaults{"min_p": float64(0.05)},
		},
		{
			name: "null primary allows alias",
			data: `{"repetition_penalty": null, "repeat_penalty": 1.1, "repeat_last_n": null, "penalty_last_n": 128}`,
			want: GenerationDefaults{"repeat_penalty": float64(1.1), "repeat_last_n": int64(128)},
		},
		{
			name: "null aliases allow final alias",
			data: `{"repetition_penalty": null, "repeat_penalty": null, "penalty_repeat": 1.2}`,
			want: GenerationDefaults{"repeat_penalty": float64(1.2)},
		},
		{
			name: "explicit zero takes precedence",
			data: `{"temperature": 0, "top_p": 0, "top_k": 0, "repetition_penalty": 0, "repeat_penalty": 1.1, "repeat_last_n": 0, "penalty_last_n": 128}`,
			want: GenerationDefaults{
				"temperature": float64(0), "top_p": float64(0), "top_k": int64(0),
				"repeat_penalty": float64(0), "repeat_last_n": int64(0),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseHFGenerationDefaults([]byte(tt.data))
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("defaults = %#v, want %#v", got, tt.want)
			}
		})
	}
}
