package envconfig

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseUint(t *testing.T) {
	tests := []struct {
		in   string
		want int
	}{
		{"0", 0},
		{"1", 1},
		{"123", 123},
		{"", -1},
		{" 42 ", 42},
		{"-1", -1},
		{"abc", -1},
		{"12x", -1},
	}
	for _, tt := range tests {
		if got := parseUint(tt.in); got != tt.want {
			t.Fatalf("parseUint(%q) = %d; want %d", tt.in, got, tt.want)
		}
	}
}

func TestParseUintList(t *testing.T) {
	tests := []struct {
		in   string
		want []int
	}{
		{"", nil},
		{"1", []int{1}},
		{"1,2,3", []int{1, 2, 3}},
		{" 4 , 5 , 6 ", []int{4, 5, 6}},
		{"1, -2", nil}, // invalid -> whole list rejected
		{"a,b", nil},
	}
	for _, tt := range tests {
		got := parseUintList(tt.in)
		if (got == nil) != (tt.want == nil) {
			t.Fatalf("parseUintList(%q) = %v; want %v", tt.in, got, tt.want)
		}
		if got == nil {
			continue
		}
		if len(got) != len(tt.want) {
			t.Fatalf("parseUintList(%q) len=%d; want %d", tt.in, len(got), len(tt.want))
		}
		for i := range got {
			if got[i] != tt.want[i] {
				t.Fatalf("parseUintList(%q)[%d]=%d; want %d", tt.in, i, got[i], tt.want[i])
			}
		}
	}
}

func TestLoadOverride_Basic(t *testing.T) {
	dir := t.TempDir()
	cfg := filepath.Join(dir, "over.ini")
	content := `
; comment
[llama3.2-vision:90b]
n-gpu-layers=33
tensor-split=10,20,30

[other]
n-gpu-layers=1
`
	if err := os.WriteFile(cfg, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_OVERRIDE_CONFIG", cfg)

	ovr := LoadOverride("llama3.2-vision:90b")
	if ovr == nil {
		t.Fatalf("LoadOverride returned nil")
	}
	if ovr.ModelName != "llama3.2-vision:90b" {
		t.Fatalf("ModelName=%q; want %q", ovr.ModelName, "llama3.2-vision:90b")
	}
	// n-gpu-layers must be the sum of tensor-split entries (10+20+30=60).
	if ovr.NumGPULayers != 60 {
		t.Fatalf("NumGPULayers=%d; want %d", ovr.NumGPULayers, 60)
	}
	wantSplit := []int{10, 20, 30}
	if len(ovr.TensorSplit) != len(wantSplit) {
		t.Fatalf("TensorSplit len=%d; want %d", len(ovr.TensorSplit), len(wantSplit))
	}
	for i := range wantSplit {
		if ovr.TensorSplit[i] != wantSplit[i] {
			t.Fatalf("TensorSplit[%d]=%d; want %d", i, ovr.TensorSplit[i], wantSplit[i])
		}
	}
}

func TestLoadOverride_NoMatchOrEmpty(t *testing.T) {
	dir := t.TempDir()
	cfg := filepath.Join(dir, "over.ini")
	content := `
[some-model]
n-gpu-layers=7
`
	if err := os.WriteFile(cfg, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_OVERRIDE_CONFIG", cfg)

	// Section exists but different model -> nil
	if got := LoadOverride("another-model"); got != nil {
		t.Fatalf("expected nil for unmatched section, got %#v", got)
	}

	// File missing -> nil
	t.Setenv("OLLAMA_OVERRIDE_CONFIG", filepath.Join(dir, "missing.ini"))
	if got := LoadOverride("some-model"); got != nil {
		t.Fatalf("expected nil for missing file, got %#v", got)
	}
}

func TestLoadOverride_BadValuesIgnored(t *testing.T) {
	dir := t.TempDir()
	cfg := filepath.Join(dir, "over.ini")
	content := `
[m]
n-gpu-layers=abc
tensor-split=1,2,x
`
	if err := os.WriteFile(cfg, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_OVERRIDE_CONFIG", cfg)

	if got := LoadOverride("m"); got != nil {
		t.Fatalf("expected nil when no valid keys parsed, got %#v", got)
	}
}
