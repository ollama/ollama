package llm

import (
	"embed"
	"log"
	"os"
	"strings"
)

//go:embed llama.cpp/build/*/*/lib/*.so
var libEmbed embed.FS

func updatePath(dir string) {
	pathComponents := strings.Split(os.Getenv("LD_LIBRARY_PATH"), ":")
	for _, comp := range pathComponents {
		if comp == dir {
			return
		}
	}
	newPath := strings.Join(append([]string{dir}, pathComponents...), ":")
	log.Printf("Updating LD_LIBRARY_PATH to %s", newPath)
	os.Setenv("LD_LIBRARY_PATH", newPath)
}
