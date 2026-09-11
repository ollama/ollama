package llm

import (
	"bytes"
	"io"
	"log/slog"
	"regexp"
	"strings"

	"github.com/ollama/ollama/envconfig"
)

// runnerLogFilter drops llama-server's routine per-request logging before it
// reaches Ollama's own log output.
//
// llama-server runs at --log-verbosity 4 (TRACE) because the startup
// memory/offload lines Ollama parses for scheduler accounting are emitted at
// llama.cpp's INFO level, and the per-request chatter is split across both
// INFO and TRACE. That means no single verbosity setting both keeps the lines
// the scheduler needs and drops the noise, so the selection happens here
// instead.
//
// This filter is deliberately the last writer in the chain
// (memoryParsingWriter -> StatusWriter -> runnerLogFilter -> stderr): every
// parser upstream still observes the unfiltered stream, so filtering cannot
// affect memory accounting, offload detection, or captured error text.
//
// Suppression is an explicit list of known-routine messages rather than a
// structural rule. Warnings and errors share the same "slot ... | task N |"
// shape as the routine lines, and --no-log-prefix strips the level marker, so
// matching on shape alone could swallow a diagnostic. With an explicit list an
// unrecognized line always passes through: if a llama.cpp update renames a
// message the noise returns, which is self-correcting, rather than a warning
// going missing.
type runnerLogFilter struct {
	out io.Writer

	// llama-server prints sampler parameters as a header line followed by an
	// indented block. When the header is dropped its continuation lines, which
	// carry no header of their own, are dropped with it.
	inSamplerParams bool
}

func newRunnerLogFilter(out io.Writer) *runnerLogFilter {
	return &runnerLogFilter{out: out}
}

// slotLineRegex matches llama-server's per-request slot logging, which is
// formatted as "slot %12s: id %2d | task %d | <message>", and captures the
// message. The task id is negative for slot bookkeeping that runs between
// requests.
var slotLineRegex = regexp.MustCompile(`^slot\s+\S+:\s+id\s+\d+\s+\|\s+task\s+-?\d+\s+\|\s*(.*)$`)

// srvLineRegex matches llama-server's server-scope logging, formatted as
// "srv  %12s: <message>", and captures the message.
var srvLineRegex = regexp.MustCompile(`^srv\s+\S+:\s*(.*)$`)

// routineRunnerMessages are llama-server messages emitted once or more per
// request. Each entry is matched as a prefix of the message body, after the
// "slot"/"srv" header is stripped and surrounding space is trimmed. The
// strings are taken from the SLT_INF/SLT_TRC/SRV_TRC call sites in
// llama.cpp's tools/server/server-context.cpp.
var routineRunnerMessages = []string{
	// Slot selection and task lifecycle.
	"- checking sim =",
	"- copying state to child",
	"- skipping, is_processing =",
	"- skipping, slot is empty",
	"selected slot by LCP similarity",
	"selected slot by LRU",
	"selected slot by id",
	"launching slots for parent task",
	"processing task, is_child =",
	"stop processing:",
	"all slots are idle",

	// Sampler setup.
	"sampler chain:",
	"sampler params:",
	"init sampler, took",

	// Prompt and KV cache bookkeeping.
	"new prompt, n_ctx_slot =",
	"prompt processing, n_tokens =",
	"cached n_tokens =",
	"clearing prompt with",
	"checking checkpoint with",
	"restored context checkpoint",
	"erased invalidated context checkpoint",
	"erasing context checkpoint too close",
	"forcing full prompt re-processing",
	"reusing chunk with size",
	"encoding mtmd batch from idx =",
	"saving idle slot to prompt cache",
	"updating prompt cache",
	"prompt cache update took",
	"- saving prompt with length",

	// Per-request timings and speculative-decoding stats.
	"prompt eval time =",
	"eval time =",
	"total time =",
	"graphs reused =",
	"n_gen =",
	"accepted",
	"draft acceptance =",
	"acc per pos =",
}

// suppressRunnerLine reports whether a llama-server log line is routine
// per-request output that should be kept out of Ollama's log.
func suppressRunnerLine(line string) bool {
	var body string
	switch match := slotLineRegex.FindStringSubmatch(line); {
	case match != nil:
		body = match[1]
	default:
		match = srvLineRegex.FindStringSubmatch(line)
		if match == nil {
			return false
		}
		body = match[1]
	}

	// Timing lines are padded for column alignment, so compare against the
	// trimmed body.
	body = strings.TrimSpace(body)
	matched := false
	for _, msg := range routineRunnerMessages {
		if strings.HasPrefix(body, msg) {
			matched = true
			break
		}
	}
	if !matched {
		return false
	}

	// Never drop anything Ollama already recognizes as an error. This reuses
	// the detection in status.go so error prefixes and out-of-memory messages
	// survive regardless of which slot or task emitted them.
	return statusErrorLine(line) == ""
}

// isSamplerParamsHeader reports whether a line introduces the indented sampler
// parameter block.
func isSamplerParamsHeader(line string) bool {
	match := slotLineRegex.FindStringSubmatch(line)
	if match == nil {
		return false
	}
	return strings.HasPrefix(strings.TrimSpace(match[1]), "sampler params:")
}

// Write forwards b to the underlying writer with routine per-request lines
// removed. Like StatusWriter, it assumes llama-server writes whole lines; a
// line split across two calls is classified on the fragment it arrives in.
func (w *runnerLogFilter) Write(b []byte) (int, error) {
	if w.out == nil {
		return len(b), nil
	}

	// OLLAMA_DEBUG asks for everything the runner emits.
	if envconfig.LogLevel() <= slog.LevelDebug {
		return w.out.Write(b)
	}

	var kept bytes.Buffer
	lastIdx := bytes.Count(b, []byte{'\n'})
	for i, raw := range bytes.Split(b, []byte{'\n'}) {
		// The final element after a trailing newline is empty and carries no
		// line of its own.
		last := i == lastIdx
		if last && len(raw) == 0 {
			continue
		}

		line := strings.TrimRight(string(raw), " \t\r")

		// Continuation lines of a dropped sampler parameter block are indented
		// and have no header to match on.
		if w.inSamplerParams {
			if line == "" || strings.HasPrefix(line, " ") || strings.HasPrefix(line, "\t") {
				continue
			}
			w.inSamplerParams = false
		}

		if suppressRunnerLine(line) {
			if isSamplerParamsHeader(line) {
				w.inSamplerParams = true
			}
			continue
		}

		kept.Write(raw)
		if !last {
			kept.WriteByte('\n')
		}
	}

	if kept.Len() > 0 {
		if _, err := w.out.Write(kept.Bytes()); err != nil {
			return 0, err
		}
	}

	// Report the full input as written so callers copying into this writer
	// don't treat suppressed lines as a short write.
	return len(b), nil
}
