package create

import (
	"strings"
	"testing"
)

func TestAutoRendererParserNotice(t *testing.T) {
	tests := []struct {
		name     string
		arch     string
		renderer string
		parser   string
		note     string
		want     string
	}{
		{
			name:     "same renderer and parser",
			arch:     "nemotron_h",
			renderer: "nemotron-3-nano",
			parser:   "nemotron-3-nano",
			want:     `using autodetected renderer/parser "nemotron-3-nano" for architecture "nemotron_h"`,
		},
		{
			name:     "different renderer and parser",
			arch:     "Qwen3ForCausalLM",
			renderer: "qwen3-coder",
			parser:   "qwen3",
			want:     `using autodetected renderer "qwen3-coder" and parser "qwen3" for architecture "Qwen3ForCausalLM"`,
		},
		{
			name:   "parser only",
			arch:   "test",
			parser: "qwen3",
			want:   `using autodetected parser "qwen3" for architecture "test"`,
		},
		{
			name:     "template note appended",
			arch:     "nemotron_h",
			renderer: "nemotron-3-nano",
			parser:   "nemotron-3-nano",
			note:     GGUFTemplateBypassed,
			want:     `using autodetected renderer/parser "nemotron-3-nano" for architecture "nemotron_h"; ` + GGUFTemplateBypassed,
		},
		{name: "nothing inferred", arch: "test", want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := AutoRendererParserNotice(tt.arch, tt.renderer, tt.parser, tt.note)
			if got != tt.want {
				t.Errorf("AutoRendererParserNotice() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestAutoStopTokensNotice(t *testing.T) {
	if got := AutoStopTokensNotice([]string{"<turn|>"}); got != `using autodetected stop token "<turn|>"` {
		t.Errorf("AutoStopTokensNotice() = %q", got)
	}
	if got := AutoStopTokensNotice(nil); got != "" {
		t.Errorf("AutoStopTokensNotice(nil) = %q, want empty", got)
	}
}

func TestMissingSamplerDefaultNoticeText(t *testing.T) {
	got := missingSamplerDefaultNotice("temperature", 0.8)
	want := "warning: temperature unset; using default temperature=0.8 (set PARAMETER temperature N to change)"
	if got != want {
		t.Errorf("missingSamplerDefaultNotice() = %q, want %q", got, want)
	}
}

func TestDraftNotices(t *testing.T) {
	tests := []struct {
		name          string
		parameters    map[string]any
		embeddedDraft bool
		distinctDraft bool
		wantContains  string
	}{
		{name: "no draft", wantContains: ""},
		{name: "embedded draft without depth", embeddedDraft: true, wantContains: "drafting model detected"},
		{
			name:          "embedded draft with depth set",
			embeddedDraft: true,
			parameters:    map[string]any{"draft_num_predict": 8},
			wantContains:  "",
		},
		{name: "distinct draft reports default depth", distinctDraft: true, wantContains: "using default draft_num_predict=4"},
		{
			name:          "distinct draft hides embedded warning",
			embeddedDraft: true,
			distinctDraft: true,
			wantContains:  "using default draft_num_predict=4",
		},
		{
			name:          "distinct draft with explicit depth",
			distinctDraft: true,
			parameters:    map[string]any{"draft_num_predict": 2},
			wantContains:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			notices := DraftNotices(tt.parameters, tt.embeddedDraft, tt.distinctDraft)
			if tt.wantContains == "" {
				if len(notices) != 0 {
					t.Fatalf("DraftNotices() = %v, want none", notices)
				}
				return
			}
			if len(notices) != 1 || !strings.Contains(notices[0], tt.wantContains) {
				t.Fatalf("DraftNotices() = %v, want one notice containing %q", notices, tt.wantContains)
			}
		})
	}
}

func TestSamplerDefaultsNotice(t *testing.T) {
	notices := SamplerDefaultsNotice(nil, nil)
	if len(notices) != 2 {
		t.Fatalf("SamplerDefaultsNotice(nil, nil) = %v, want one line each for temperature and top_p", notices)
	}

	notices = SamplerDefaultsNotice(map[string]any{"temperature": 0.5}, nil)
	if len(notices) != 1 || !strings.Contains(notices[0], "top_p unset") {
		t.Fatalf("SamplerDefaultsNotice(temperature set) = %v, want only top_p notice", notices)
	}

	notices = SamplerDefaultsNotice(nil, map[string]any{"top_p": 0.95})
	if len(notices) != 1 || !strings.Contains(notices[0], "temperature unset") {
		t.Fatalf("SamplerDefaultsNotice(source top_p) = %v, want only temperature notice", notices)
	}

	notices = SamplerDefaultsNotice(map[string]any{"temperature": 0.5}, map[string]any{"top_p": 0.95})
	if len(notices) != 0 {
		t.Fatalf("SamplerDefaultsNotice(both set) = %v, want none", notices)
	}
}

func TestMissingRequiresNotice(t *testing.T) {
	got := MissingRequiresNotice("0.19.0")
	want := "warning: no REQUIRES; using default 0.19.0 (set REQUIRES to change)"
	if got != want {
		t.Errorf("MissingRequiresNotice() = %q, want %q", got, want)
	}
}
