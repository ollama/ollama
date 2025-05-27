package server

import (
	"testing"
	"text/template"
)

func TestExtractThinking(t *testing.T) {
	tests := []struct {
		in, wantContent, wantThink string
	}{
		{
			in:          "<think> internal </think> world",
			wantThink:   "internal ",
			wantContent: "world",
		},
		{
			in:          "<think>a</think><think>b</think>c",
			wantThink:   "a",
			wantContent: "<think>b</think>c",
		},
		{
			in:          "no think",
			wantThink:   "",
			wantContent: "no think",
		},
	}
	for i, tt := range tests {
		parser := thinkingParser{
			openingTag: "<think>",
			closingTag: "</think>",
		}
		gotThinking, gotContent := parser.addContent(tt.in)
		if gotContent != tt.wantContent || gotThinking != tt.wantThink {
			t.Errorf("case %d: got (%q,%q), want (%q,%q)", i, gotThinking, gotContent, tt.wantThink, tt.wantContent)
		}
	}
}

func TestThinkingStreaming(t *testing.T) {
	type step struct {
		input          string
		wantThinking   string
		wantContent    string
		wantStateAfter thinkingState
	}

	cases := []struct {
		desc  string
		skip  bool
		steps []step
	}{
		{
			desc: "content without a thinking tag",
			steps: []step{
				{
					input:          "  abc",
					wantThinking:   "",
					wantContent:    "  abc",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
		{
			desc: "content before a thinking tag nerfs the thinking tag",
			steps: []step{
				{
					input:          "  abc <think>def</think> ghi",
					wantThinking:   "",
					wantContent:    "  abc <think>def</think> ghi",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
		{
			desc: "building up a thinking tag partially",
			// skip: true,
			steps: []step{
				{
					input:          "  <th",
					wantThinking:   "",
					wantContent:    "",
					wantStateAfter: thinkingState_LookingForOpening,
				},
				{
					input:          "in",
					wantThinking:   "",
					wantContent:    "",
					wantStateAfter: thinkingState_LookingForOpening,
				},
				{
					input:          "k>a",
					wantThinking:   "a",
					wantContent:    "",
					wantStateAfter: thinkingState_Thinking,
				},
			},
		},
		{
			desc: "partial closing tag",
			steps: []step{
				{
					input:          "<think>abc</th",
					wantThinking:   "abc",
					wantContent:    "",
					wantStateAfter: thinkingState_Thinking,
				},
				{
					input:          "ink>def",
					wantThinking:   "",
					wantContent:    "def",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
		{
			desc: "partial closing tag fakeout",
			steps: []step{
				{
					input:          "<think>abc</th",
					wantThinking:   "abc",
					wantContent:    "",
					wantStateAfter: thinkingState_Thinking,
				},
				{
					input:          "ing>def",
					wantThinking:   "</thing>def",
					wantContent:    "",
					wantStateAfter: thinkingState_Thinking,
				},
				{
					input:          "ghi</thi",
					wantThinking:   "ghi",
					wantContent:    "",
					wantStateAfter: thinkingState_Thinking,
				},
				{
					input:          "nk>jkl",
					wantThinking:   "",
					wantContent:    "jkl",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
		{
			desc: "whitespace after thinking tag",
			steps: []step{
				{
					input:          "  <think>abc</think>\n\ndef",
					wantThinking:   "abc",
					wantContent:    "def",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
		{
			desc: "whitespace after thinking tag (incremental)",
			steps: []step{
				{
					input:          "  <think>abc</think>",
					wantThinking:   "abc",
					wantContent:    "",
					wantStateAfter: thinkingState_ThinkingDoneEatingWhitespace,
				},
				{
					input:          "\n\ndef",
					wantThinking:   "",
					wantContent:    "def",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
		{
			desc: "whitespace after thinking tag with content and more whitespace",
			steps: []step{
				{
					input:          "  <think>abc</think>\n\ndef ",
					wantThinking:   "abc",
					wantContent:    "def ",
					wantStateAfter: thinkingState_ThinkingDone,
				},
				{
					input:          " ghi",
					wantThinking:   "",
					wantContent:    " ghi",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
		{
			desc: "token by token",
			steps: []step{
				{
					input:          "<think>",
					wantThinking:   "",
					wantContent:    "",
					wantStateAfter: thinkingState_ThinkingStartedEatingWhitespace,
				},
				{
					input:          "\n",
					wantThinking:   "",
					wantContent:    "",
					wantStateAfter: thinkingState_ThinkingStartedEatingWhitespace,
				},
				{
					input:          "</think>",
					wantThinking:   "",
					wantContent:    "",
					wantStateAfter: thinkingState_ThinkingDoneEatingWhitespace,
				},
				{
					input:          "\n\n",
					wantThinking:   "",
					wantContent:    "",
					wantStateAfter: thinkingState_ThinkingDoneEatingWhitespace,
				},
				{
					input:          "Hi",
					wantThinking:   "",
					wantContent:    "Hi",
					wantStateAfter: thinkingState_ThinkingDone,
				},
				{
					input:          " there",
					wantThinking:   "",
					wantContent:    " there",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
		{
			desc: "leading thinking whitespace",
			steps: []step{
				{
					input:          "  <think>   \t ",
					wantThinking:   "",
					wantContent:    "",
					wantStateAfter: thinkingState_ThinkingStartedEatingWhitespace,
				},
				{
					input:          "  these are some ",
					wantThinking:   "these are some ",
					wantContent:    "",
					wantStateAfter: thinkingState_Thinking,
				},
				{
					input:          "thoughts </think>  ",
					wantThinking:   "thoughts ",
					wantContent:    "",
					wantStateAfter: thinkingState_ThinkingDoneEatingWhitespace,
				},
				{
					input:          "  more content",
					wantThinking:   "",
					wantContent:    "more content",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
	}

	for _, c := range cases {
		parser := thinkingParser{
			openingTag: "<think>",
			closingTag: "</think>",
		}
		if c.skip {
			continue
		}
		for i, step := range c.steps {
			thinking, content := parser.addContent(step.input)
			if content != step.wantContent || thinking != step.wantThinking {
				t.Errorf("case %q (step %d): got (%q,%q), want (%q,%q)", c.desc, i, content, thinking, step.wantContent, step.wantThinking)
			}
			if parser.state != step.wantStateAfter {
				t.Errorf("case %q (step %d): got state %s, want %s", c.desc, i, parser.state, step.wantStateAfter)
			}
		}
	}
}

func TestInferThinkingTags(t *testing.T) {
	cases := []struct {
		desc           string
		tmplString     string
		wantOpeningTag string
		wantClosingTag string
	}{
		{
			desc: "basic",
			tmplString: `
			{{ if .Thinking}}
				/think
			{{ end }}
			{{- range $i, $_ := .Messages }}
				{{- $last := eq (len (slice $.Messages $i)) 1 -}}
				{{ if and $last .Thinking }}
					<think>{{ .Thinking }}</think>
				{{ end }}
			{{ end }}
		`,
			wantOpeningTag: "<think>",
			wantClosingTag: "</think>",
		},
		{
			desc: "doubly nested range",
			tmplString: `
			{{ if .Thinking}}
				/think
			{{ end }}
			{{- range $i, $_ := .Messages }}
				{{- range $j, $_ := .NotMessages }}
					{{- $last := eq (len (slice $.Messages $i)) 1 -}}
					{{ if and $last .Thinking }}
						<think>{{ .Thinking }}</think>
					{{ end }}
				{{ end }}
			{{ end }}
		`,
			wantOpeningTag: "",
			wantClosingTag: "",
		},
		{
			desc: "whitespace is trimmed",
			tmplString: `
			{{ if .Thinking}}
				/think
			{{ end }}
			{{- range $i, $_ := .Messages }}
				{{- $last := eq (len (slice $.Messages $i)) 1 -}}
				{{ if and $last .Thinking }}
					Some text before   {{ .Thinking }}    Some text after
				{{ end }}
			{{ end }}
		`,
			wantOpeningTag: "Some text before",
			wantClosingTag: "Some text after",
		},
		{
			desc: "qwen3",
			tmplString: `
{{- if or .System .Tools .Thinking }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}
{{- if .Thinking }}
/think
{{- else }}
/no_think
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if and $last .Thinking }}
<think>{{ .Thinking }}</think>
{{ end }}
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
			`,
			wantOpeningTag: "<think>",
			wantClosingTag: "</think>",
		},
	}
	for _, c := range cases {
		tmpl := template.Must(template.New("test").Parse(c.tmplString))
		openingTag, closingTag := inferThinkingTags(tmpl)
		if openingTag != c.wantOpeningTag || closingTag != c.wantClosingTag {
			t.Errorf("case %q: got (%q,%q), want (%q,%q)", c.desc, openingTag, closingTag, c.wantOpeningTag, c.wantClosingTag)
		}
	}
}
