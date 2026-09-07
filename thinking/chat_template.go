package thinking

import "strings"

// chatTemplateThinkTags are the opening/closing pairs a GGUF chat template can
// render a reasoning trace between.
//
// Spelled out as a closed table rather than matched by a pattern: a tag pair is
// only usable if the template renders *both* halves, and a heuristic that
// guessed a closing tag would hand the sampler a cut point that never arrives.
// Order matters only in that the first pair a template carries wins, and no
// template in practice carries two.
var chatTemplateThinkTags = []struct{ open, close string }{
	{"<think>", "</think>"},
	{"<|channel>", "<channel|>"},
	{"<|START_THINKING|>", "<|END_THINKING|>"},
	{"<seed:think>", "</seed:think>"},
}

// InferTagsFromChatTemplate infers the tags a GGUF chat template renders
// reasoning between, for models served through that template rather than
// through a renderer or a Go template.
//
// [InferTags] can only answer for a Go template: it walks the parsed AST for a
// range over Messages that references Thinking, and a Jinja chat template has
// no such AST here — the runner applies it, so the server never parses it. That
// left a model whose reasoning tags are perfectly well known — they appear
// verbatim in its own template — with no tags at all, and a thinking budget is
// dropped when the tags are unknown. So the budget silently did nothing for
// every model served by its embedded chat template, including ones that asked
// for a budget explicitly on the request.
//
// The check is deliberately literal. The tags have to appear in the template
// text because that is what the template will emit, so containment is evidence
// rather than a guess, and requiring both halves means a template that only
// mentions an opening tag contributes nothing.
func InferTagsFromChatTemplate(chatTemplate string) (string, string) {
	if chatTemplate == "" {
		return "", ""
	}

	for _, pair := range chatTemplateThinkTags {
		if strings.Contains(chatTemplate, pair.open) && strings.Contains(chatTemplate, pair.close) {
			return pair.open, pair.close
		}
	}

	// Some Qwen/DeepSeek templates strip prior reasoning by splitting assistant
	// content at the closing tag and never write the opening one, so the pair
	// is not both-present above. The model still emits both — the split is only
	// possible because it does — and the server already reads these templates as
	// thinking-capable for exactly this shape, so refusing to name the tags here
	// would leave that family the one case that declares thinking and cannot be
	// budgeted.
	for _, pair := range chatTemplateThinkTags {
		if strings.Contains(chatTemplate, "content.split('"+pair.close+"')") ||
			strings.Contains(chatTemplate, `content.split("`+pair.close+`")`) {
			return pair.open, pair.close
		}
	}

	return "", ""
}
