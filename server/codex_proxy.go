package server

import (
	"log/slog"
	"net/http"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/proxy"
)

const (
	codexDesktopChatGPTURL  = "https://chatgpt.com/backend-api/codex"
	codexDesktopOpenAIURL   = "https://api.openai.com/v1"
	codexDesktopLogFilename = "codex-proxy.log"
)

func newCodexDesktopProxy() (http.Handler, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		slog.Warn("failed to find home directory for Codex proxy", "error", err)
		home = os.TempDir()
	}

	return proxy.NewCodexDesktop(proxy.CodexDesktopConfig{
		OllamaURL:          envconfig.ConnectableHost().String(),
		ChatGPTURL:         codexDesktopChatGPTURL,
		OpenAIURL:          codexDesktopOpenAIURL,
		RoutingCatalogPath: filepath.Join(home, ".codex", proxy.CodexDesktopRoutingCatalogFilename),
		ActivityLogPath:    filepath.Join(home, ".ollama", "logs", codexDesktopLogFilename),
		Logger:             slog.Default(),
	})
}
