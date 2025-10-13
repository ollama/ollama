package renderers

import "github.com/ollama/ollama/api"

type Renderer interface {
	Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error)
}

func RendererForName(name string) Renderer {
	switch name {
	case "qwen3-coder":
		renderer := &Qwen3CoderRenderer{}
		return renderer
	case "qwen3-vl":
		renderer := &Qwen3VLRenderer{false}
		return renderer
	default:
		return nil
	}
}
