package renderers

import "github.com/ollama/ollama/api"

// type rendererFunc func([]api.Message, []api.Tool, *api.ThinkValue) (string, error)

// func RenderWithRenderer(name string, msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
// 	renderer := rendererForName(name)
// 	if renderer == nil {
// 		return "", fmt.Errorf("unknown renderer %q", name)
// 	}
// 	return renderer(msgs, tools, think)
// }

type Renderer interface {
	Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error)
}

// func rendererForName(name string) rendererFunc {
func RendererForName(name string) Renderer {
	switch name {
	case "qwen3-coder":
		renderer := &Qwen3CoderRenderer{false} // this is not implemented yet
		return renderer
	case "qwen3-vl":
		renderer := &Qwen3VLRenderer{false}
		return renderer
	default:
		return nil
	}
}
