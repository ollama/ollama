package main

import (
    "github.com/ollama/ollama/app/lifecycle"
)

var RunFunc = lifecycle.Run

func main() {
    RunFunc()
}
