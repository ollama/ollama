package llm

import (
	"embed"
	"log"
	"os"
	"strings"
)

//go:embed llama.cpp/gguf/build/windows/*/lib/*.dll
var libEmbed embed.FS

func updatePath(dir string) {
	pathComponents := strings.Split(os.Getenv("PATH"), ";")
	for _, comp := range pathComponents {
		// Case incensitive
		if strings.ToLower(comp) == strings.ToLower(dir) {
			return
		}
	}
	newPath := strings.Join(append(pathComponents, dir), ";")
	log.Printf("Updating PATH to %s", newPath)
	os.Setenv("PATH", newPath)
}

func verifyDriverAccess() error {
	// TODO if applicable
	return nil
}
