package main

import (
	"context"

	"github.com/jmorganca/ollama/cmd"
)

func main() {
	cmd.NewCLI().ExecuteContext(context.Background())
}
