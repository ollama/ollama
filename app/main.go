package main

// Compile with the following to get rid of the cmd pop up on windows
// go build -ldflags="-H windowsgui" .

import (
	"os"

	"github.com/jmorganca/ollama/app/lifecycle"
)

func main() {
	// TODO - remove as we end the early access phase
	os.Setenv("OLLAMA_DEBUG", "1") // nolint:errcheck

	lifecycle.Run()
}
