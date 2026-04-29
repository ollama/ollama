package llm

import (
	"bytes"
	"io"
	"strings"
	"sync/atomic"
)

// StatusWriter is a writer that captures error messages from the llama runner process
type StatusWriter struct {
	out io.Writer
	// Subprocess wrappers may wire both stdout and stderr to the same
	// StatusWriter, and os/exec serializes Write calls in that case.
	lastErrMsg atomic.Value
}

const maxCapturedErrorBytes = 8 * 1024

func NewStatusWriter(out io.Writer) *StatusWriter {
	return &StatusWriter{
		out: out,
	}
}

func (w *StatusWriter) LastError() string {
	if w == nil {
		return ""
	}
	if v := w.lastErrMsg.Load(); v != nil {
		return v.(string)
	}
	return ""
}

func (w *StatusWriter) SetLastError(msg string) {
	if w == nil {
		return
	}
	w.lastErrMsg.Store(msg)
}

func (w *StatusWriter) AppendError(msg string) {
	if w == nil || msg == "" {
		return
	}

	if current := w.LastError(); current != "" {
		msg = current + "\n" + msg
	}

	if len(msg) > maxCapturedErrorBytes {
		msg = msg[len(msg)-maxCapturedErrorBytes:]
		if i := strings.IndexByte(msg, '\n'); i >= 0 {
			msg = msg[i+1:]
		}
	}

	w.SetLastError(msg)
}

// TODO - regex matching to detect errors like
// libcublasLt.so.11: cannot open shared object file: No such file or directory
// TODO - if we later see error lines split across multiple Write calls in real
// logs, add a small rolling buffer here to capture those fragments.

var errorPrefixes = []string{
	"error:",
	"CUDA error",
	"ROCm error",
	"cudaMalloc failed",
	"\"ERR\"",
	"error loading model",
	"GGML_ASSERT",
	"Deepseek2 does not support K-shift",
	"signal arrived during cgo execution",
	"llama_init_from_model:",
}

var outOfMemorySubstrings = []string{
	"out of memory",
	"out of device memory",
	"cudaMalloc failed",
	"hipMalloc failed",
	"failed to allocate",
	"allocation failed",
	"not enough memory",
	"insufficient memory",
	"vk_error_out_of_device_memory",
	"erroroutofmemory",
}

func IsOutOfMemory(err error) bool {
	if err == nil {
		return false
	}
	return IsOutOfMemoryMessage(err.Error())
}

func IsOutOfMemoryMessage(msg string) bool {
	msg = strings.ToLower(msg)
	for _, needle := range outOfMemorySubstrings {
		if strings.Contains(msg, strings.ToLower(needle)) {
			return true
		}
	}
	return false
}

func (w *StatusWriter) Write(b []byte) (int, error) {
	for _, raw := range bytes.Split(b, []byte{'\n'}) {
		line := strings.TrimRight(string(raw), " \t\r")
		if line == "" {
			continue
		}

		if errMsg := statusErrorLine(line); errMsg != "" {
			w.AppendError(errMsg)
		}
	}

	if w.out == nil {
		return len(b), nil
	}

	return w.out.Write(b)
}

func statusErrorLine(line string) string {
	for _, prefix := range errorPrefixes {
		if _, after, ok := strings.Cut(line, prefix); ok {
			return prefix + strings.TrimRight(after, " \t\r")
		}
	}

	if IsOutOfMemoryMessage(line) {
		return line
	}

	return ""
}
