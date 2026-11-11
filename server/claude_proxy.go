package server

import (
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/claudedesktop"
	"github.com/ollama/ollama/internal/proxy"
)

func newClaudeDesktopProxy() (*proxy.ClaudeDesktop, error) {
	endpoint := envconfig.ConnectableHost()
	resolver := claudedesktop.NewResolver(endpoint)
	return proxy.NewClaudeDesktop(proxy.ClaudeDesktopConfig{
		PathPrefix:         proxy.ClaudeDesktopPathPrefix,
		OllamaURL:          endpoint.String(),
		RefreshModels:      resolver.Models,
		ResolveAccessState: resolver.AccessState,
		ListLocalModels:    resolver.LocalModels,
	})
}
