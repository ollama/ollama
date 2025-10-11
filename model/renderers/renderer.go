package renderers

import (
	"fmt"

	"github.com/ollama/ollama/api"
)

type rendererFunc func([]api.Message, []api.Tool, *api.ThinkValue) (string, error)

type RendererRegistry struct {
	renderers map[string]rendererFunc
}

func (r *RendererRegistry) Register(name string, renderer rendererFunc) {
	r.renderers[name] = renderer
}

var registry = RendererRegistry{
	renderers: make(map[string]rendererFunc),
}

func Register(name string, renderer rendererFunc) {
	registry.Register(name, renderer)
}

func RenderWithRenderer(name string, msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	renderer := rendererForName(name)
	if renderer == nil {
		return "", fmt.Errorf("unknown renderer %q", name)
	}
	return renderer(msgs, tools, think)
}

func rendererForName(name string) rendererFunc {
	if renderer, ok := registry.renderers[name]; ok {
		return renderer
	}
	switch name {
	case "qwen3-coder":
		return Qwen3CoderRenderer
	default:
		return nil
	}
}
