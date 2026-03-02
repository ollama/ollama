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
