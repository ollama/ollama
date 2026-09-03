package server

import (
	"log/slog"
	"net/http"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/app/codexproxy"
	"github.com/ollama/ollama/envconfig"
)

const (
	codexProxyChatGPTURL  = "https://chatgpt.com/backend-api/codex"
	codexProxyOpenAIURL   = "https://api.openai.com/v1"
	codexProxyLogFilename = "codex-proxy.log"
)

func newCodexProxyHandler() (http.Handler, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		slog.Warn("failed to find home directory for Codex proxy", "error", err)
		home = os.TempDir()
	}

	return codexproxy.New(codexproxy.Config{
		PathPrefix:         codexproxy.PathPrefix,
		OllamaURL:          envconfig.ConnectableHost().String(),
		ChatGPTURL:         codexProxyChatGPTURL,
		OpenAIURL:          codexProxyOpenAIURL,
		RoutingCatalogPath: filepath.Join(home, ".codex", codexproxy.RoutingCatalogFilename),
		ActivityLogPath:    filepath.Join(home, ".ollama", "logs", codexProxyLogFilename),
		Logger:             slog.Default(),
	})
}
