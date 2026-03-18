package main

import (
	"fmt"
	"os"

	"github.com/ollama/ollama/x/ortrunner/oga"
)

func main() {
	if err := oga.CheckInit(); err != nil {
		fmt.Fprintf(os.Stderr, "ORT GenAI init failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("ORT GenAI dynamic library loaded successfully!")
	fmt.Println("All symbols resolved.")
}
