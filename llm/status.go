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
	"cudaMalloc failed",
	"\"ERR\"",
}

func (w *StatusWriter) Write(b []byte) (int, error) {
	var errMsg string
	for _, prefix := range errorPrefixes {
		if _, after, ok := bytes.Cut(b, []byte(prefix)); ok {
			errMsg = prefix + string(bytes.TrimSpace(after))
		}
	}

	if bytes.Contains(b, []byte("unknown model architecture")) {
		if _, after, ok := bytes.Cut(b, []byte("architecture")); ok {
			errMsg = "error" + string(bytes.TrimSpace(after))

			if before, _, ok := bytes.Cut(after, []byte("llama_load")); ok {
				errMsg = "error" + string(bytes.TrimSpace(before))
			}

			errMsg = errMsg + "\nYour current version of Ollama doesn't support this model architecture. Consider upgrading."
		}
	}

	if errMsg != "" {
		w.LastErrMsg = errMsg
	}

	return w.out.Write(b)
}
