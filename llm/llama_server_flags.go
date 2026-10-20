package llm

import (
	"context"
	"log/slog"
	"os"
	"os/exec"
	"regexp"
	"sync"
	"time"
)

// llama.cpp renamed --webui/--no-webui to --ui/--no-ui and documents the
// old spelling as deprecated. The vendored llama.cpp (LLAMA_CPP_VERSION b11724-unsloth-ds41,
// and upstream b11521 before it) accepts both, so --no-ui is the default. A
// runner directory can still carry an older llama-server (for example a
// FETCHCONTENT_SOURCE_DIR_LLAMA_CPP or OLLAMA_LLAMA_CPP_SOURCE build), and
// the Go side has no compiled-in llama.cpp version, so the binary's own
// --help output decides; anything unexpected falls back to --no-webui, which
// every llama-server version still accepts.
const (
	llamaServerNoUIFlag       = "--no-ui"
	llamaServerLegacyNoUIFlag = "--no-webui"
	llamaServerHelpTimeout    = 10 * time.Second
)

var noUIHelpRegex = regexp.MustCompile(`(?m)(^|[\s,])--no-ui([\s,]|$)`)

type llamaServerBinaryKey struct {
	path    string
	size    int64
	modTime time.Time
}

var (
	llamaServerNoUIFlags sync.Map // llamaServerBinaryKey -> string

	// llamaServerHelpFn is replaced in tests.
	llamaServerHelpFn = func(exe string) ([]byte, error) {
		ctx, cancel := context.WithTimeout(context.Background(), llamaServerHelpTimeout)
		defer cancel()
		cmd := exec.CommandContext(ctx, exe, "--help")
		SetupLlamaServerCommandEnv(cmd, exe, nil, nil)
		return cmd.CombinedOutput()
	}
)

// LlamaServerNoUIFlag returns the flag that disables llama-server's web UI
// for the binary at exe. The result is cached per binary path, size and
// modification time, so a rebuilt binary is probed again.
func LlamaServerNoUIFlag(exe string) string {
	info, err := os.Stat(exe)
	if err != nil {
		return llamaServerLegacyNoUIFlag
	}
	key := llamaServerBinaryKey{path: exe, size: info.Size(), modTime: info.ModTime()}
	if v, ok := llamaServerNoUIFlags.Load(key); ok {
		return v.(string)
	}

	help, err := llamaServerHelpFn(exe)
	flag := noUIFlagFromHelp(help, err)
	if err != nil {
		slog.Debug("llama-server --help failed; using legacy web UI flag", "exe", exe, "error", err)
	}
	slog.Debug("llama-server web UI flag", "exe", exe, "flag", flag)
	llamaServerNoUIFlags.Store(key, flag)
	return flag
}

func noUIFlagFromHelp(help []byte, err error) string {
	if err == nil && noUIHelpRegex.Match(help) {
		return llamaServerNoUIFlag
	}
	return llamaServerLegacyNoUIFlag
}
