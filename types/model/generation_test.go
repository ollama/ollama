package model

import "testing"

func TestParseHFGenerationDefaults(t *testing.T) {
	defaults, err := ParseHFGenerationDefaults([]byte(`{
		"top_k": 0,
		"top_p": 0.7,
		"min_p": 0,
		"temperature": 0.6,
		"repetition_penalty": 1.05,
		"penalty_last_n": -1
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

	check("top_k", int64(0))
	check("top_p", float64(0.7))
	check("min_p", float64(0))
	check("temperature", float64(0.6))
	check("repeat_penalty", float64(1.05))
	check("repeat_last_n", int64(-1))
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
