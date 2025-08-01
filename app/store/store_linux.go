package store

import (
	"os"
	"path/filepath"

	"github.com/ollama/ollama/envconfig"
)

func getStorePath() string {
	if os.Geteuid() == 0 {
		// TODO where should we store this on linux for system-wide operation?
		return "/etc/ollama/config.json"
	}

	return filepath.Join(envconfig.ConfigDir(), "config.json")
}
