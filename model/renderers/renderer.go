package renderers

import (
	"fmt"

	"github.com/ollama/ollama/api"
)

type rendererFunc func([]api.Message, []api.Tool, *api.ThinkValue) (string, error)

func RenderWithRenderer(name string, msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	renderer := rendererForName(name)
	if renderer == nil {
		return "", fmt.Errorf("unknown renderer %q", name)
	}
	return renderer(msgs, tools, think)
}

func rendererForName(name string) rendererFunc {
	switch name {
	case "qwen3-coder":
		return Qwen3CoderRenderer
	default:
		return nil
	}
}
