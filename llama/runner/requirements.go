package main

import (
	"encoding/json"
	"os"

	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/version"
)

func printRequirements(fp *os.File) {
	attrs := map[string]string{
		"system_info":  llama.PrintSystemInfo(),
		"version":      version.Version,
		"cpu_features": llama.CpuFeatures,
	}
	enc := json.NewEncoder(fp)
	_ = enc.Encode(attrs)
}
