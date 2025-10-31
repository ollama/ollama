package llm

import (
	"bytes"
	"os"
	"regexp"
)

// StatusWriter is a writer that captures error messages from the llama runner process
type StatusWriter struct {
	LastErrMsg string
	out        *os.File
}

func NewStatusWriter(out *os.File) *StatusWriter {
	return &StatusWriter{
		out: out,
	}
}

// Error string prefixes
var errorPrefixes = []string {
	"error:",
	"CUDA error",
	"ROCm error",
	"cudaMalloc failed",
	"\"ERR\"",
	"error loading model",
	"GGML_ASSERT",
	"Deepseek2 does not support K-shift",
}
// List of regex patterns to catch errors without prefixes
var errorPatterns = []*regexp.Regexp {

	// Shared library loading errors (CUDA, ROCm, system libraries)
	regexp.MustCompile(`lib[\w.-]+\.so[.\d]*: cannot open shared object file`),
	regexp.MustCompile(`lib[\w.-]+\.dylib.*: .*not found`), // macOS dylib errors

	// Memory allocation failures
	regexp.MustCompile(`failed to allocate [\d.]+ [KMGT]?i?B`),
	regexp.MustCompile(`out of memory|OOM`),
	regexp.MustCompile(`cudaMalloc failed`),
	regexp.MustCompile(`hipMalloc failed`), // AMD ROCm

	// GPU/Device errors
	regexp.MustCompile(`CUDA error \d+`),
	regexp.MustCompile(`ROCm error`),
	regexp.MustCompile(`Metal error`),
	regexp.MustCompile(`failed to (initialize|create) (CUDA|ROCm|Metal|Vulkan) (device|context|backend)`),

	// Model file errors
	regexp.MustCompile(`(invalid|unsupported|unknown) model (format|version|architecture)`),
	regexp.MustCompile(`failed to (load|read|open) model`),
	regexp.MustCompile(`(corrupted|invalid) (tensor|weight|layer)`),

	// Assertion failures with context
	regexp.MustCompile(`GGML_ASSERT:.*\.(c|cpp|cu|metal):\d+`),

	// Backend initialization failures
	regexp.MustCompile(`ggml_backend.*failed`),
	regexp.MustCompile(`failed to initialize.*backend`),
}

func (w *StatusWriter) Write(b []byte) (int, error) {
	var errMsg string

	// First check simple prefixes
	for _, prefix := range errorPrefixes {
		if _, after, ok := bytes.Cut(b, []byte(prefix)); ok {
			errMsg = prefix + string(bytes.TrimSpace(after))
			break
		}
	}

	// Check regex patterns if no prefix match
	if errMsg == "" {
		for _, pattern := range errorPatterns {
			if match := pattern.Find(b); match != nil {
				// Capture the matched error and some surrounding context
				start := pattern.FindIndex(b)
				if start != nil {
					// Include up to 200 chars of context around the match
					contextStart := start[0]
					if contextStart > 50 {
						contextStart -= 50
					} else {
						contextStart = 0
					}
					contextEnd := start[1] + 150
					if contextEnd > len(b) {
						contextEnd = len(b)
					}
					errMsg = string(bytes.TrimSpace(b[contextStart:contextEnd]))
					break
				}
			}
		}
	}

	if errMsg != "" {
		w.LastErrMsg = errMsg
	}

	return w.out.Write(b)
}
