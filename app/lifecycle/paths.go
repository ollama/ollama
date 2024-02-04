package lifecycle

import (
	"os"
	"path/filepath"
	"runtime"
)

var (
	AppName    = "ollama app"
	CLIName    = "ollama"
	AppDir     = "/opt/Ollama"
	AppDataDir = "/opt/Ollama"
	// TODO - should there be a distinct log dir?
	UpdateStageDir = "/tmp"
	ServerLogFile  = "/tmp/ollama.log"
	Installer      = "Ollama Setup.exe"
)

func init() {
	if runtime.GOOS == "windows" {
		AppName += ".exe"
		CLIName += ".exe"
		// Logs, configs, downloads go to LOCALAPPDATA
		localAppData := os.Getenv("LOCALAPPDATA")
		AppDataDir = filepath.Join(localAppData, "Ollama")
		UpdateStageDir = filepath.Join(AppDataDir, "updates")
		ServerLogFile = filepath.Join(AppDataDir, "server.log")

		// Executables are stored in APPDATA
		AppDir = filepath.Join(localAppData, "Programs", "Ollama")

	} else if runtime.GOOS == "darwin" {
		// TODO
		AppName += ".app"
		// } else if runtime.GOOS == "linux" {
		// TODO
	}
}
