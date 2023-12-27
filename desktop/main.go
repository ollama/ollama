package main

// Compile with the following to get rid of the cmd pop up on windows
// go build -ldflags="-H windowsgui" .

import (
	"github.com/jmorganca/ollama/desktop/lifecycle"
)

func main() {
	lifecycle.Run()
}
