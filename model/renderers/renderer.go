package renderers

import (
	"fmt"

	"github.com/ollama/ollama/api"
)

type Renderer interface {
	Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error)
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
	case "functiongemma":
		return &FunctionGemmaRenderer{}
	default:
		return nil
	}
}
