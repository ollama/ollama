package llm

import (
	"bytes"
	"io"
	"maps"
	"strings"
	"testing"
)

// runnerLogSample is the llama-server output reported in ollama/ollama#16897,
// interleaved with the startup lines Ollama parses for scheduler accounting.
const runnerLogSample = `using device CUDA0 (NVIDIA GeForce RTX 4060 Ti) (0000:01:00.0) - 15221 MiB free
load_tensors: offloaded 33/33 layers to GPU
load_tensors: CUDA0 model buffer size =   852.89 MiB
load_tensors: CPU_Mapped model buffer size =   308.23 MiB
llama_context: CUDA0 KV buffer size =  1920.00 MiB
llama_context: CUDA0 compute buffer size =   378.04 MiB
slot get_availabl: id  4 | task -1 | selected slot by LCP similarity, sim_best = 1.000 (> 0.100 thold), f_keep = 0.651
slot launch_slot_: id  4 | task -1 | sampler chain: logits -> penalties -> dist
slot launch_slot_: id  4 | task -1 | sampler params:
	repeat_last_n = 64, repeat_penalty = 1.100, frequency_penalty = 0.000
	top_k = 40, top_p = 1.000, min_p = 0.000, temp = 0.000
slot launch_slot_: id  4 | task 3794 | processing task, is_child = 0
slot process_sing: id  0 | task -1 | saving idle slot to prompt cache
slot update_slots: id  4 | task 3794 | new prompt, n_ctx_slot = 28160, n_keep = 4, task.n_tokens = 69
slot update_slots: id  4 | task 3794 | restored context checkpoint (pos_min = 0, pos_max = 64, n_tokens = 65)
slot update_slots: id  4 | task 3794 | cached n_tokens = 64, memory_seq_rm [64, end)
slot init_sampler: id  4 | task 3794 | init sampler, took 0.04 ms, tokens: text = 69, total = 69
slot print_timing: id  4 | task 3794 | prompt eval time =      23.60 ms /     5 tokens
slot print_timing: id  4 | task 3794 |        eval time =     217.72 ms /    38 tokens
slot print_timing: id  4 | task 3794 |       total time =     241.33 ms /    43 tokens
slot print_timing: id  4 | task 3794 |    graphs reused =       3750
slot      release: id  4 | task 3794 | stop processing: n_tokens = 106, truncated = 0
srv  update_slots: all slots are idle
`

func filterLog(t *testing.T, in string) string {
	t.Helper()

	var buf bytes.Buffer
	w := newRunnerLogFilter(&buf)
	n, err := w.Write([]byte(in))
	if err != nil {
		t.Fatal(err)
	}
	if n != len(in) {
		t.Fatalf("Write reported %d bytes, want %d", n, len(in))
	}
	return buf.String()
}

func TestRunnerLogFilterDropsPerRequestLines(t *testing.T) {
	got := filterLog(t, runnerLogSample)

	for _, noise := range []string{
		"selected slot by LCP similarity",
		"sampler chain:",
		"sampler params:",
		"repeat_last_n = 64",
		"top_k = 40",
		"processing task, is_child",
		"saving idle slot to prompt cache",
		"new prompt, n_ctx_slot",
		"restored context checkpoint",
		"cached n_tokens",
		"init sampler, took",
		"prompt eval time",
		"eval time",
		"total time",
		"graphs reused",
		"stop processing",
		"all slots are idle",
	} {
		if strings.Contains(got, noise) {
			t.Errorf("kept per-request line containing %q\n%s", noise, got)
		}
	}
}

// The scheduler parses these lines out of llama-server's output, so the filter
// must never remove them.
func TestRunnerLogFilterKeepsSchedulerLines(t *testing.T) {
	got := filterLog(t, runnerLogSample)

	for _, want := range []string{
		"using device CUDA0 (NVIDIA GeForce RTX 4060 Ti) (0000:01:00.0) - 15221 MiB free",
		"load_tensors: offloaded 33/33 layers to GPU",
		"load_tensors: CUDA0 model buffer size =   852.89 MiB",
		"load_tensors: CPU_Mapped model buffer size =   308.23 MiB",
		"llama_context: CUDA0 KV buffer size =  1920.00 MiB",
		"llama_context: CUDA0 compute buffer size =   378.04 MiB",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("dropped scheduler line %q\n%s", want, got)
		}
	}
}

func TestRunnerLogFilterKeepsLines(t *testing.T) {
	tests := []struct {
		name string
		log  string
	}{
		{
			name: "error wearing a slot header",
			log:  "slot update_slots: id  4 | task 12 | error: failed to decode batch\n",
		},
		{
			name: "out of memory wearing a slot header",
			log:  "slot update_slots: id  4 | task 12 | CUDA error: out of memory\n",
		},
		{
			name: "unrecognized slot message",
			log:  "slot update_slots: id  4 | task 12 | something new upstream added\n",
		},
		{
			name: "slot warning",
			log:  "slot update_slots: id  0 | task 5 | slot context shift is disabled\n",
		},
		{
			name: "server startup",
			log:  "srv          init: initializing, n_slots = 4, n_ctx_slot = 28160\n",
		},
		{
			name: "model load",
			log:  "srv    load_model: loading model 'qwen3.5:4b'\n",
		},
		{
			name: "assertion",
			log:  "GGML_ASSERT(buffer) failed\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := filterLog(t, tt.log); got != tt.log {
				t.Errorf("got %q, want %q", got, tt.log)
			}
		})
	}
}

func TestRunnerLogFilterDebugKeepsEverything(t *testing.T) {
	t.Setenv("OLLAMA_DEBUG", "1")

	if got := filterLog(t, runnerLogSample); got != runnerLogSample {
		t.Errorf("OLLAMA_DEBUG=1 should pass everything through, got:\n%s", got)
	}
}

func TestRunnerLogFilterSamplerParamsBlockEndsAtNextHeader(t *testing.T) {
	const log = `slot launch_slot_: id  4 | task -1 | sampler params:
	top_k = 40, top_p = 1.000
llama_context: CUDA0 KV buffer size =  1920.00 MiB
`
	got := filterLog(t, log)
	want := "llama_context: CUDA0 KV buffer size =  1920.00 MiB\n"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// llama-server may emit several lines in one write, and the indented sampler
// block can arrive in a later call than its header.
func TestRunnerLogFilterAcrossWrites(t *testing.T) {
	var buf bytes.Buffer
	w := newRunnerLogFilter(&buf)

	for _, chunk := range []string{
		"slot launch_slot_: id  4 | task -1 | sampler params: \n",
		"\ttop_k = 40, top_p = 1.000\n\tmirostat = 0\n",
		"load_tensors: offloaded 33/33 layers to GPU\n",
	} {
		if _, err := w.Write([]byte(chunk)); err != nil {
			t.Fatal(err)
		}
	}

	want := "load_tensors: offloaded 33/33 layers to GPU\n"
	if got := buf.String(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestRunnerLogFilterPreservesPartialLine(t *testing.T) {
	const log = "load_tensors: offloaded 33/33 layers to GPU"
	if got := filterLog(t, log); got != log {
		t.Errorf("got %q, want %q", got, log)
	}
}

// The filter sits at the end of the writer chain, so replaying a log through
// the real chain must parse exactly the same memory values as parsing it with
// the filter absent.
func TestRunnerLogFilterDoesNotAffectMemoryAccounting(t *testing.T) {
	parse := func(out io.Writer) *llamaServerRunner {
		t.Helper()

		runner := &llamaServerRunner{
			vramByDevice:     make(map[string]uint64),
			systemFreeAtLoad: make(map[string]uint64),
		}
		w := &memoryParsingWriter{inner: NewStatusWriter(out), runner: runner}
		if _, err := w.Write([]byte(runnerLogSample)); err != nil {
			t.Fatal(err)
		}
		return runner
	}

	unfiltered := parse(io.Discard)

	var buf bytes.Buffer
	filtered := parse(newRunnerLogFilter(&buf))

	if filtered.memTotal != unfiltered.memTotal {
		t.Errorf("memTotal = %d, want %d", filtered.memTotal, unfiltered.memTotal)
	}
	if filtered.memGPU != unfiltered.memGPU {
		t.Errorf("memGPU = %d, want %d", filtered.memGPU, unfiltered.memGPU)
	}
	if filtered.gpuLayers != unfiltered.gpuLayers || filtered.totalLayers != unfiltered.totalLayers {
		t.Errorf("layers = %d/%d, want %d/%d",
			filtered.gpuLayers, filtered.totalLayers, unfiltered.gpuLayers, unfiltered.totalLayers)
	}
	if !maps.Equal(filtered.vramByDevice, unfiltered.vramByDevice) {
		t.Errorf("vramByDevice = %v, want %v", filtered.vramByDevice, unfiltered.vramByDevice)
	}
	if !maps.Equal(filtered.systemFreeAtLoad, unfiltered.systemFreeAtLoad) {
		t.Errorf("systemFreeAtLoad = %v, want %v", filtered.systemFreeAtLoad, unfiltered.systemFreeAtLoad)
	}

	// Guard against the sample silently stopping to exercise the parser.
	if unfiltered.memTotal == 0 || unfiltered.gpuLayers == 0 || len(unfiltered.vramByDevice) == 0 {
		t.Fatal("sample log did not exercise the memory parser")
	}
	if strings.Contains(buf.String(), "all slots are idle") {
		t.Error("filter did not run in the chain")
	}
}

// The memory parser and error capture run upstream of the filter, so they must
// still observe suppressed lines.
func TestStatusWriterSeesSuppressedLines(t *testing.T) {
	var buf bytes.Buffer
	status := NewStatusWriter(newRunnerLogFilter(&buf))

	const log = "slot update_slots: id  4 | task 12 | cached n_tokens = 64, memory_seq_rm [64, end)\n" +
		"slot update_slots: id  4 | task 12 | CUDA error: out of memory\n"

	if _, err := status.Write([]byte(log)); err != nil {
		t.Fatal(err)
	}

	if got := status.LastError(); !strings.Contains(got, "CUDA error: out of memory") {
		t.Errorf("error not captured, got %q", got)
	}
	if got := buf.String(); strings.Contains(got, "cached n_tokens") {
		t.Errorf("routine line reached output: %q", got)
	}
}
