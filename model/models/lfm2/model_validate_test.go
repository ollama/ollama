package lfm2

import "testing"

func TestModelValidateMissingTokenEmbedding(t *testing.T) {
	m := &Model{}

	err := m.Validate()
	if err == nil {
		t.Fatal("expected validation error")
	}
	if got, want := err.Error(), "lfm2: missing token_embd tensor"; got != want {
		t.Fatalf("Validate() error = %q, want %q", got, want)
	}
}
