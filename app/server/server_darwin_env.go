//go:build darwin

package server

import (
	"bytes"
	"context"
	"os/exec"
	"strings"
	"sync"
	"time"
)

var launchctlGetenv = getenvFromLaunchctl

type launchctlResult struct {
	value string
	found bool
}

var launchctlCache sync.Map

var darwinLaunchctlEnvAllowlist = []string{
	"GGML_METAL_TENSOR_DISABLE",
	"OLLAMA_DEBUG",
	"OLLAMA_DEBUG_LOG_REQUESTS",
	"OLLAMA_FLASH_ATTENTION",
	"OLLAMA_GPU_OVERHEAD",
	"OLLAMA_KEEP_ALIVE",
	"OLLAMA_KV_CACHE_TYPE",
	"OLLAMA_LLM_LIBRARY",
	"OLLAMA_LOAD_TIMEOUT",
	"OLLAMA_MAX_LOADED_MODELS",
	"OLLAMA_MAX_QUEUE",
	"OLLAMA_NUM_PARALLEL",
	"OLLAMA_SCHED_SPREAD",
	"OLLAMA_MULTIUSER_CACHE",
	"OLLAMA_NEW_ENGINE",
}

func mergeDarwinLaunchctlEnv(env map[string]string) {
	for _, key := range darwinLaunchctlEnvAllowlist {
		if _, ok := env[key]; ok {
			continue
		}
		if value, ok := launchctlGetenv(key); ok {
			env[key] = value
		}
	}
}

func getenvFromLaunchctl(key string) (string, bool) {
	if cached, ok := launchctlCache.Load(key); ok {
		r := cached.(launchctlResult)
		return r.value, r.found
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "launchctl", "getenv", key)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	var result launchctlResult
	if out, err := cmd.Output(); err == nil {
		if value := strings.TrimSpace(string(out)); value != "" {
			result = launchctlResult{value: value, found: true}
		}
	}
	launchctlCache.Store(key, result)
	return result.value, result.found
}
