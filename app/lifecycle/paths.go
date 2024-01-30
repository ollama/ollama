package lifecycle

import (
	"os"
	"path/filepath"
	"runtime"
)

var (
	AppName        = "ollama app"
	CLIName        = "ollama"
	AppDir         = "/opt/Ollama"
	UpdateStageDir = "/tmp"
	ServerLogFile  = "/tmp/ollama.log"
	Installer      = "Ollama Setup.exe"
)

func init() {
	if runtime.GOOS == "windows" {
		AppName += ".exe"
		CLIName += ".exe"
		localAppData := os.Getenv("LOCALAPPDATA")
		AppDir = filepath.Join(localAppData, "Ollama")
		UpdateStageDir = filepath.Join(AppDir, "updates")
		ServerLogFile = filepath.Join(AppDir, "server.log")
	} else if runtime.GOOS == "darwin" {
		// TODO
		AppName += ".app"
		// } else if runtime.GOOS == "linux" {
		// TODO
	}
}
