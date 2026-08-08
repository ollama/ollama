package launch

import (
	"fmt"
	"net/url"
	"os"
	"strings"

	"github.com/ollama/ollama/envconfig"
)

const (
	webBrainLaunchURL          = "https://webbrain.one/launch/ollama"
	webBrainDefaultContextSize = 65536
)

// WebBrain implements Runner for the WebBrain browser extension handoff.
type WebBrain struct{}

func (w *WebBrain) String() string { return "WebBrain" }

func (w *WebBrain) Run(model string, models []LaunchModel, args []string) error {
	if len(args) > 0 {
		return fmt.Errorf("webbrain does not support extra args")
	}
	if strings.TrimSpace(model) == "" {
		return fmt.Errorf("webbrain requires a model")
	}

	handoff := webBrainHandoffURL(model, models)
	fmt.Fprintf(os.Stderr, "\nOpening WebBrain Ollama setup:\n  %s\n\n", hyperlink(handoff, handoff))
	fmt.Fprintf(os.Stderr, "Confirm the WebBrain prompt in your browser to use %s via Ollama.\n\n", model)
	OpenBrowser(handoff)
	return nil
}

func webBrainHandoffURL(model string, models []LaunchModel) string {
	u, _ := url.Parse(webBrainLaunchURL)
	q := u.Query()
	q.Set("source", "ollama")
	q.Set("model", model)
	q.Set("baseUrl", webBrainOllamaBaseURL())
	q.Set("contextWindow", fmt.Sprintf("%d", webBrainContextWindow(models)))
	u.RawQuery = q.Encode()
	return u.String()
}

func webBrainOllamaBaseURL() string {
	u := envconfig.ConnectableHost()
	clone := *u
	clone.RawQuery = ""
	clone.Fragment = ""
	clone.Path = strings.TrimRight(clone.Path, "/") + "/v1"
	return clone.String()
}

func webBrainContextWindow(models []LaunchModel) int {
	if len(models) > 0 && models[0].ContextLength > 0 {
		return models[0].ContextLength
	}
	return webBrainDefaultContextSize
}
