package llm

import (
	"bytes"
	"os"
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

// TODO - regex matching to detect errors like
// libcublasLt.so.11: cannot open shared object file: No such file or directory

var errorPrefixes = []string{
	"error:",
	"CUDA error",
	"ROCm error",
	"cudaMalloc failed",
	"\"ERR\"",
	"error loading model",
	"GGML_ASSERT",
	"Deepseek2 does not support K-shift",
}

func (w *StatusWriter) Write(b []byte) (int, error) {
	var errMsg string
	for _, prefix := range errorPrefixes {
		if _, after, ok := bytes.Cut(b, []byte(prefix)); ok {
			errMsg = prefix + string(bytes.TrimSpace(after))
		}
	}
	if errMsg != "" {
		w.LastErrMsg = errMsg
	}

	return w.out.Write(b)
}
