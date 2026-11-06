package thinking

import "testing"

func TestInferTagsFromChatTemplate(t *testing.T) {
	cases := []struct {
		name     string
		template string
		open     string
		close    string
	}{
		{
			name:     "empty template has no tags",
			template: "",
		},
		{
			// The shape Ornith-1.5 and the qwen3.5 family ship: the tags are
			// written literally into the assistant branch.
			name: "qwen-style think tags",
			template: `{%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}
    {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
{%- endif %}`,
			open:  "<think>",
			close: "</think>",
		},
		{
			name:     "gemma4 channel tags",
			template: `{{- '<|channel>' + thought + '<channel|>' }}`,
			open:     "<|channel>",
			close:    "<channel|>",
		},
		{
			name:     "cohere thinking tags",
			template: `{{ '<|START_THINKING|>' + reasoning + '<|END_THINKING|>' }}`,
			open:     "<|START_THINKING|>",
			close:    "<|END_THINKING|>",
		},
		{
			// Half a pair is not a pair: a budget needs somewhere to cut, and a
			// closing tag that never arrives is worse than no budget at all.
			name:     "opening tag alone is not enough",
			template: `{{- '<think>' + reasoning }}`,
		},
		{
			name:     "closing tag alone is not enough",
			template: `{{- reasoning + '</think>' }}`,
		},
		{
			name:     "a template with no reasoning tags",
			template: `{%- for message in messages %}{{ message.content }}{%- endfor %}`,
		},
		{
			// The Qwen/DeepSeek shape the capability check already treats as
			// thinking-capable: the closing tag appears only as a split point,
			// and the opening tag is never written by the template at all.
			name:     "reasoning stripped by splitting on the closing tag",
			template: `{%- set content = message.content.split('</think>')[-1].lstrip('\n') %}`,
			open:     "<think>",
			close:    "</think>",
		},
		{
			name:     "same shape with double quotes",
			template: `{%- set content = message.content.split("</think>")[-1] %}`,
			open:     "<think>",
			close:    "</think>",
		},
		{
			// A bare mention of a closing tag is not a split, and inventing a
			// pair from it would hand the sampler a cut point the model never
			// reaches.
			name:     "closing tag mentioned but not used as a split point",
			template: `{# the model may emit </think> here #}{{ message.content }}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			open, close := InferTagsFromChatTemplate(tc.template)
			if open != tc.open || close != tc.close {
				t.Errorf("InferTagsFromChatTemplate() = (%q, %q), want (%q, %q)", open, close, tc.open, tc.close)
			}
		})
	}
}
