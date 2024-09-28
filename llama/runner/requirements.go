package main

import (
	"encoding/json"
	"os"
	"strings"

	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/version"
)

func printRequirements(fp *os.File) {
	// TODO formal type...
	attrs := map[string]interface{}{
		"system_info":  llama.PrintSystemInfo(),
		"version":      version.Version,
		"cpu_features": strings.Split(llama.CpuFeatures, ","),
	}
	enc := json.NewEncoder(fp)
	_ = enc.Encode(attrs)
}
