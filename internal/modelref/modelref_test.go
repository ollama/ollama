package modelref

import (
	"errors"
	"testing"
)

func TestParseRef(t *testing.T) {
	tests := []struct {
		name         string
		input        string
		wantBase     string
		wantSource   ModelSource
		wantErr      error
		wantCloud    bool
		wantLocal    bool
		wantStripped string
		wantStripOK  bool
	}{
		{
			name:         "cloud suffix",
			input:        "gpt-oss:20b:cloud",
			wantBase:     "gpt-oss:20b",
			wantSource:   ModelSourceCloud,
			wantCloud:    true,
			wantStripped: "gpt-oss:20b",
			wantStripOK:  true,
		},
		{
			name:         "legacy cloud suffix",
			input:        "gpt-oss:20b-cloud",
			wantBase:     "gpt-oss:20b",
			wantSource:   ModelSourceCloud,
			wantCloud:    true,
			wantStripped: "gpt-oss:20b",
			wantStripOK:  true,
		},
		{
			name:         "local suffix",
			input:        "qwen3:8b:local",
			wantBase:     "qwen3:8b",
			wantSource:   ModelSourceLocal,
			wantLocal:    true,
			wantStripped: "qwen3:8b:local",
		},
		{
			name:         "no source suffix",
			input:        "llama3.2",
			wantBase:     "llama3.2",
			wantSource:   ModelSourceUnspecified,
			wantStripped: "llama3.2",
		},
		{
			name:         "bare cloud name is not explicit cloud",
			input:        "my-cloud-model",
			wantBase:     "my-cloud-model",
			wantSource:   ModelSourceUnspecified,
			wantStripped: "my-cloud-model",
		},
		{
			name:         "slash in suffix blocks legacy cloud parsing",
			input:        "foo:bar-cloud/baz",
			wantBase:     "foo:bar-cloud/baz",
			wantSource:   ModelSourceUnspecified,
			wantStripped: "foo:bar-cloud/baz",
		},
		{
			name:       "conflicting source suffixes",
			input:      "foo:cloud:local",
			wantErr:    ErrConflictingSourceSuffix,
			wantSource: ModelSourceUnspecified,
		},
		{
			name:       "empty input",
			input:      "   ",
			wantErr:    ErrModelRequired,
			wantSource: ModelSourceUnspecified,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseRef(tt.input)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("ParseRef(%q) error = %v, want %v", tt.input, err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseRef(%q) returned error: %v", tt.input, err)
			}

			if got.Base != tt.wantBase {
				t.Fatalf("base = %q, want %q", got.Base, tt.wantBase)
			}

			if got.Source != tt.wantSource {
				t.Fatalf("source = %v, want %v", got.Source, tt.wantSource)
			}

			if HasExplicitCloudSource(tt.input) != tt.wantCloud {
				t.Fatalf("HasExplicitCloudSource(%q) = %v, want %v", tt.input, HasExplicitCloudSource(tt.input), tt.wantCloud)
			}

			if HasExplicitLocalSource(tt.input) != tt.wantLocal {
				t.Fatalf("HasExplicitLocalSource(%q) = %v, want %v", tt.input, HasExplicitLocalSource(tt.input), tt.wantLocal)
			}

			stripped, ok := StripCloudSourceTag(tt.input)
			if ok != tt.wantStripOK {
				t.Fatalf("StripCloudSourceTag(%q) ok = %v, want %v", tt.input, ok, tt.wantStripOK)
			}
			if stripped != tt.wantStripped {
				t.Fatalf("StripCloudSourceTag(%q) base = %q, want %q", tt.input, stripped, tt.wantStripped)
			}
		})
	}
}

func TestNormalizePullName(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantName  string
		wantCloud bool
		wantErr   error
	}{
		{
			name:     "explicit local strips source",
			input:    "gpt-oss:20b:local",
			wantName: "gpt-oss:20b",
		},
		{
			name:      "explicit cloud with size maps to legacy dash cloud tag",
			input:     "gpt-oss:20b:cloud",
			wantName:  "gpt-oss:20b-cloud",
			wantCloud: true,
		},
		{
			name:      "legacy cloud with size remains stable",
			input:     "gpt-oss:20b-cloud",
			wantName:  "gpt-oss:20b-cloud",
			wantCloud: true,
		},
		{
			name:      "explicit cloud without tag maps to cloud tag",
			input:     "qwen3:cloud",
			wantName:  "qwen3:cloud",
			wantCloud: true,
		},
		{
			name:      "host port without tag keeps host port and appends cloud tag",
			input:     "localhost:11434/library/foo:cloud",
			wantName:  "localhost:11434/library/foo:cloud",
			wantCloud: true,
		},
		{
			name:    "conflicting source suffixes fail",
			input:   "foo:cloud:local",
			wantErr: ErrConflictingSourceSuffix,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotName, gotCloud, err := NormalizePullName(tt.input)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("NormalizePullName(%q) error = %v, want %v", tt.input, err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("NormalizePullName(%q) returned error: %v", tt.input, err)
			}

			if gotName != tt.wantName {
				t.Fatalf("normalized name = %q, want %q", gotName, tt.wantName)
			}
			if gotCloud != tt.wantCloud {
				t.Fatalf("cloud = %v, want %v", gotCloud, tt.wantCloud)
			}
		})
	}
}

func TestParseSourceSuffix(t *testing.T) {
	tests := []struct {
		name         string
		input        string
		wantBase     string
		wantSource   ModelSource
		wantExplicit bool
	}{
		{
			name:         "explicit cloud suffix",
			input:        "gpt-oss:20b:cloud",
			wantBase:     "gpt-oss:20b",
			wantSource:   ModelSourceCloud,
			wantExplicit: true,
		},
		{
			name:         "explicit local suffix",
			input:        "qwen3:8b:local",
			wantBase:     "qwen3:8b",
			wantSource:   ModelSourceLocal,
			wantExplicit: true,
		},
		{
			name:         "legacy cloud suffix on tag",
			input:        "gpt-oss:20b-cloud",
			wantBase:     "gpt-oss:20b",
			wantSource:   ModelSourceCloud,
			wantExplicit: true,
		},
		{
			name:         "legacy cloud suffix does not match model segment",
			input:        "my-cloud-model",
			wantBase:     "my-cloud-model",
			wantSource:   ModelSourceUnspecified,
			wantExplicit: false,
		},
		{
			name:         "legacy cloud suffix blocked when suffix includes slash",
			input:        "foo:bar-cloud/baz",
			wantBase:     "foo:bar-cloud/baz",
			wantSource:   ModelSourceUnspecified,
			wantExplicit: false,
		},
		{
			name:         "unknown suffix is not explicit source",
			input:        "gpt-oss:clod",
			wantBase:     "gpt-oss:clod",
			wantSource:   ModelSourceUnspecified,
			wantExplicit: false,
		},
		{
			name:         "uppercase suffix is accepted",
			input:        "gpt-oss:20b:CLOUD",
			wantBase:     "gpt-oss:20b",
			wantSource:   ModelSourceCloud,
			wantExplicit: true,
		},
		{
			name:         "no suffix",
			input:        "llama3.2",
			wantBase:     "llama3.2",
			wantSource:   ModelSourceUnspecified,
			wantExplicit: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotBase, gotSource, gotExplicit := parseSourceSuffix(tt.input)
			if gotBase != tt.wantBase {
				t.Fatalf("base = %q, want %q", gotBase, tt.wantBase)
			}
			if gotSource != tt.wantSource {
				t.Fatalf("source = %v, want %v", gotSource, tt.wantSource)
			}
			if gotExplicit != tt.wantExplicit {
				t.Fatalf("explicit = %v, want %v", gotExplicit, tt.wantExplicit)
			}
		})
	}
}
