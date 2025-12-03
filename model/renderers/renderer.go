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
	default:
		return nil
	}
}
