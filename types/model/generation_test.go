package model

import "testing"

func TestParseHFGenerationDefaults(t *testing.T) {
	defaults, err := ParseHFGenerationDefaults([]byte(`{
		"top_k": 40.0,
		"top_p": 0.7,
		"min_p": 0,
		"typical_p": 0.95,
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
	check("typical_p", float64(0.95))
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
		"mirostat_tau": 5.0
	}`))
	if err != nil {
		t.Fatal(err)
	}

	if got := defaults["top_p"]; got != float64(0.8) {
		t.Fatalf("top_p = %#v, want %#v", got, float64(0.8))
	}

	for _, key := range []string{"do_sample", "eos_token_id", "pad_token_id", "max_new_tokens", "mirostat_tau"} {
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
