package renderers

import (
	"fmt"

	"github.com/ollama/ollama/api"
)

type Renderer interface {
	Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error)
	LeadingBOS() string
}

// MessageDelimiter marks where a message of the given role starts in the
// rendered prompt; llama-server uses these to place context checkpoints at
// message boundaries.
type MessageDelimiter struct {
	Role      string `json:"role"`
	Delimiter string `json:"delimiter"`
}

// MessageDelimiterProvider is an optional interface implemented by renderers
// that use fixed, role-keyed markers to open each message. Renderers whose
// message boundaries can't be expressed as a static per-role marker should
// not implement it.
type MessageDelimiterProvider interface {
	MessageDelimiters() []MessageDelimiter
}

type (
	RendererConstructor func() Renderer
	RendererRegistry    struct {
		renderers map[string]RendererConstructor
	}
)

// RenderImgTags is a global flag that tells renderers to use [img] tags
// for images. This is set by the Ollama server package on init, or left as
// false for other environments where renderers are used
var RenderImgTags bool

func (r *RendererRegistry) Register(name string, renderer RendererConstructor) {
	r.renderers[name] = renderer
}

var registry = RendererRegistry{
	renderers: make(map[string]RendererConstructor),
}

func Register(name string, renderer RendererConstructor) {
	registry.Register(name, renderer)
}

func RenderWithRenderer(name string, msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	renderer := rendererForName(name)
	if renderer == nil {
		return "", fmt.Errorf("unknown renderer %q", name)
	}
	return renderer.Render(msgs, tools, think)
}

func LeadingBOSForRenderer(name string) string {
	renderer := rendererForName(name)
	if renderer == nil {
		return ""
	}

	return renderer.LeadingBOS()
}

// MessageDelimitersForRenderer returns the message delimiters for the named
// renderer, or nil if the renderer is unknown or doesn't have fixed,
// role-keyed message markers.
func MessageDelimitersForRenderer(name string) []MessageDelimiter {
	renderer := rendererForName(name)
	if renderer == nil {
		return nil
	}

	provider, ok := renderer.(MessageDelimiterProvider)
	if !ok {
		return nil
	}

	return provider.MessageDelimiters()
}

// chatMLMessageDelimiters returns the message delimiters shared by renderers
// that use ChatML-style "<|im_start|>role\n" markers to open messages.
func chatMLMessageDelimiters() []MessageDelimiter {
	return []MessageDelimiter{
		{Role: "system", Delimiter: imStartTag + "system\n"},
		{Role: "user", Delimiter: imStartTag + "user\n"},
		{Role: "assistant", Delimiter: imStartTag + "assistant\n"},
	}
}

func rendererForName(name string) Renderer {
	if constructor, ok := registry.renderers[name]; ok {
		return constructor()
	}
	switch name {
	case "qwen3-coder":
		renderer := &Qwen3CoderRenderer{}
		return renderer
	case "qwen3-vl-instruct":
		renderer := &Qwen3VLRenderer{isThinking: false, useImgTags: RenderImgTags}
		return renderer
	case "qwen3-vl-thinking":
		renderer := &Qwen3VLRenderer{isThinking: true, useImgTags: RenderImgTags}
		return renderer
	case "qwen3.5":
		renderer := &Qwen35Renderer{isThinking: true, emitEmptyThinkOnNoThink: true, useImgTags: RenderImgTags}
		return renderer
	case "qwen3.8":
		return newQwen38Renderer()
	case "ornith":
		return newOrnithRenderer()
	case "cogito":
		renderer := &CogitoRenderer{isThinking: true}
		return renderer
	case "deepseek3.1":
		renderer := &DeepSeek3Renderer{IsThinking: true, Variant: Deepseek31}
		return renderer
	case "olmo3":
		renderer := &Olmo3Renderer{UseExtendedSystemMessage: false}
		return renderer
	case "olmo3.1":
		renderer := &Olmo3Renderer{UseExtendedSystemMessage: true}
		return renderer
	case "olmo3-think":
		// Used for Olmo-3-7B-Think and Olmo-3.1-32B-Think (same template)
		renderer := &Olmo3ThinkRenderer{Variant: Olmo31Think}
		return renderer
	case "olmo3-32b-think":
		// Used for Olmo-3-32B-Think
		renderer := &Olmo3ThinkRenderer{Variant: Olmo3Think32B}
		return renderer
	case "nemotron-3-nano":
		return &Nemotron3NanoRenderer{}
	case "nemotron-3.5-nano":
		return &Nemotron3NanoRenderer{v35: true}
	case "gemma4", "gemma4-small":
		return &Gemma4Renderer{useImgTags: RenderImgTags}
	case "gemma4-large":
		return &Gemma4Renderer{useImgTags: RenderImgTags, emptyBlockOnNothink: true}
	case "functiongemma":
		return &FunctionGemmaRenderer{}
	case "glm-4.7":
		return &GLM47Renderer{}
	case "glm-ocr":
		return &GlmOcrRenderer{useImgTags: RenderImgTags}
	case "lfm2":
		return &LFM2Renderer{IsThinking: false, useImgTags: RenderImgTags}
	case "lfm2-thinking":
		return &LFM2Renderer{IsThinking: true, useImgTags: RenderImgTags}
	case "laguna":
		return &LagunaRenderer{}
	case "poolside-v1":
		return &LagunaV8Renderer{}
	case "cohere":
		return &CohereRenderer{}
	case "glimmer":
		return &GlimmerRenderer{useImgTags: RenderImgTags}
	default:
		return nil
	}
}
