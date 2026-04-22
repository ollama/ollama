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
	// StartRunner wires both Stdout and Stderr to the same StatusWriter, and
	// os/exec serializes Write calls in that case.
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

func (w *StatusWriter) Write(b []byte) (int, error) {
	var errMsg string
	for _, prefix := range errorPrefixes {
		if _, after, ok := bytes.Cut(b, []byte(prefix)); ok {
			line := after
			if j := bytes.IndexByte(line, '\n'); j >= 0 {
				line = line[:j]
			}
			errMsg = prefix + string(bytes.TrimRight(line, " \t\r"))
		}
	}
	if errMsg != "" {
		w.AppendError(errMsg)
	}

	return w.out.Write(b)
}
