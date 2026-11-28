package llm

import (
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

// Real output of llama-server b11963's measure role for qwen3.8-fixed
// (qwen35 27B Q4_K_M + MTP) at 64k, q8_0, ub 512, -ngl all --fit off, on
// one RTX 3090.
const qwen64kMeasureLine = `cmd_child_to_router:state:{"state":"measuring","payload":{"result":"measured","n_ctx_train":262144,"min":{"n_ctx":65536,"devices":[{"kind":"device","name":"CUDA0","model":16084563968,"context":3066167296,"compute":617912448,"total":25351356416,"free":25073352704},{"kind":"host","name":"Host","model":715161600,"context":0,"compute":109357088}]},"max":{"n_ctx":65536,"devices":[{"kind":"device","name":"CUDA0","model":16084563968,"context":3066167296,"compute":617912448,"total":25351356416,"free":25073352704},{"kind":"host","name":"Host","model":715161600,"context":0,"compute":109357088}]}}}`

func cudaGPU(id, name string, free uint64) ml.DeviceInfo {
	return ml.DeviceInfo{
		DeviceID:   ml.DeviceID{ID: id, Library: "CUDA"},
		Name:       name,
		FreeMemory: free,
	}
}

func TestParseMeasureOutput(t *testing.T) {
	out := "some log line\n" + qwen64kMeasureLine + "\n"
	p, err := parseMeasureOutput([]byte(out))
	if err != nil {
		t.Fatal(err)
	}
	if p.Max.NCtx != 65536 || len(p.Max.Devices) != 2 {
		t.Fatalf("payload = %+v", p.Max)
	}

	// Matches the logged load: CUDA0 model 15339.44 MiB, KV 2176 + RS
	// 748.12 MiB, CPU_Mapped 682.03 MiB.
	gpu := cudaGPU("GPU-1", "CUDA1", 23962*mib)
	mem, err := memoryFromPayload(p, []ml.DeviceInfo{gpu}, runnerDeviceLogNames([]ml.DeviceInfo{gpu}, map[string]string{"CUDA_VISIBLE_DEVICES": "GPU-1"}))
	if err != nil {
		t.Fatal(err)
	}
	d := mem.Devices[0]
	approxMiB(t, "model", d.Model, 15339.44)
	approxMiB(t, "context", d.Context, 2176+748.125)
	approxMiB(t, "host model", mem.HostModel, 682.03)
	if d.ID != gpu.DeviceID {
		t.Errorf("device %v, want %v", d.ID, gpu.DeviceID)
	}
	// Discovery saw 23962 MiB free; the measuring process 23911 MiB.
	if got := d.Overhead / mib; got != 50 && got != 51 {
		t.Errorf("overhead = %d MiB, want ~51", got)
	}

	for name, out := range map[string]string{
		"failed":  `cmd_child_to_router:state:{"state":"measuring","payload":{"result":"failed","error":"boom"}}`,
		"missing": "llama_model_load: error\n",
		"garbled": `cmd_child_to_router:state:{"state":`,
	} {
		if _, err := parseMeasureOutput([]byte(out)); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestMemoryFromPayloadDevices(t *testing.T) {
	p := measurePayload{Result: "measured"}
	p.Max.Devices = []measureDevice{
		{Kind: "device", Name: "CUDA0", Model: 3 << 30, Free: 20 << 30},
		{Kind: "device", Name: "CUDA1", Model: 22 << 30, Free: 23 << 30},
		{Kind: "host", Name: "Host", Model: 1 << 30},
	}
	// Discovery CUDA1 and CUDA0, launched in that order: the child renumbers
	// them CUDA0 and CUDA1.
	g1 := cudaGPU("GPU-b", "CUDA1", 20<<30)
	g0 := cudaGPU("GPU-a", "CUDA0", 30<<30)
	gpus := []ml.DeviceInfo{g1, g0}
	names := runnerDeviceLogNames(gpus, map[string]string{"CUDA_VISIBLE_DEVICES": "GPU-b,GPU-a"})
	mem, err := memoryFromPayload(p, gpus, names)
	if err != nil {
		t.Fatal(err)
	}
	if mem.Devices[0].ID != g1.DeviceID || mem.Devices[0].Model != 3<<30 || mem.Devices[1].Model != 22<<30 {
		t.Errorf("devices = %+v", mem.Devices)
	}
	// A 7 GiB gap is not the process's own overhead: capped.
	if mem.Devices[1].Overhead != maxProcessOverhead {
		t.Errorf("overhead = %d, want cap", mem.Devices[1].Overhead)
	}

	if _, err := memoryFromPayload(p, gpus[:1], names); err == nil {
		t.Error("memory on an unrequested device accepted")
	}
	if _, err := memoryFromPayload(p, []ml.DeviceInfo{cudaGPU("GPU-c", "CUDA7", 0)}, nil); err == nil {
		t.Error("missing device accepted")
	}
}

func TestMeasureArgs(t *testing.T) {
	in := []string{"--model", "m", "-c", "4096", "-ngl", "99", "--fit", "on", "-b", "512"}
	got := measureArgs(in, -1)
	want := []string{"--model", "m", "-c", "4096", "-b", "512", "-ngl", "all", "--fit", "off"}
	if !slices.Equal(got, want) {
		t.Errorf("auto: %v", got)
	}
	if got := measureArgs(in, 20); !slices.Contains(got, "20") || slices.Contains(got, "all") {
		t.Errorf("pinned num_gpu: %v", got)
	}
}

func TestFitTargets(t *testing.T) {
	t.Setenv(llamaArgFitTargetEnv, "")
	os.Unsetenv(llamaArgFitTargetEnv)
	t.Setenv(llamaArgFitEnv, "")
	os.Unsetenv(llamaArgFitEnv)

	var launch llamaServerLaunchConfig
	if got := launch.fitTargets(2); !slices.Equal(got, []uint64{1024 * mib, 1024 * mib}) {
		t.Errorf("default = %v", got)
	}
	// Ollama's projector pad: one value, broadcast.
	launch.projectors = []string{"p"}
	launch.mmprojMemory = 888 * mib
	launch.gpus = []ml.DeviceInfo{cudaGPU("GPU-1", "CUDA0", 23911*mib)}
	launch.opts = api.DefaultOptions()
	if got := launch.fitTargets(2); got[0] != 1912*mib || got[1] != 1912*mib {
		t.Errorf("projector pad = %v", got)
	}
	launch = llamaServerLaunchConfig{}
	t.Setenv(llamaArgFitTargetEnv, "2048,512")
	if got := launch.fitTargets(3); !slices.Equal(got, []uint64{2048 * mib, 512 * mib, 1024 * mib}) {
		t.Errorf("list = %v", got)
	}
	t.Setenv(llamaArgFitEnv, "off")
	if got := launch.fitTargets(2); got != nil {
		t.Errorf("fit off = %v", got)
	}
}

// TestPredictFromMeasurementMatchesFit checks the requirement against
// llama.cpp's own fit for the logged qwen 64k load (2026-11-04 09:43):
// projected 19071 MiB (main 18683 + MTP 256+132) + target 1912 (Ollama's
// pad) + 1161.02 (mmproj worst case) = 22144 MiB.
func TestPredictFromMeasurementMatchesFit(t *testing.T) {
	t.Setenv("OLLAMA_FLASH_ATTENTION", "1")
	t.Setenv("OLLAMA_KV_CACHE_TYPE", "q8_0")
	os.Unsetenv(llamaArgFitTargetEnv)
	os.Unsetenv(llamaArgFitEnv)

	p, err := parseMeasureOutput([]byte(qwen64kMeasureLine))
	if err != nil {
		t.Fatal(err)
	}
	gpu := cudaGPU("GPU-1", "CUDA0", 23911*mib)
	mem, err := memoryFromPayload(p, []ml.DeviceInfo{gpu}, nil)
	if err != nil {
		t.Fatal(err)
	}

	path, f := writeTestGGUF(t, qwen35KV(), nil)
	opts := api.DefaultOptions()
	opts.NumCtx = 65536
	opts.NumBatch = 512
	opts.DraftNumPredict = 4
	in := VRAMPredictInput{GPUs: []ml.DeviceInfo{gpu}, ModelPath: path, Model: f, Options: opts, NumParallel: 1}
	launch := llamaServerLaunchConfig{
		modelPath: path, draftType: draftTypeMTP, projectors: []string{"/proj"}, mmprojMemory: 888 * mib,
		opts: opts, numParallel: 1, gpus: in.GPUs,
	}

	recordMMProjUsage("/proj", 1161.02)
	t.Cleanup(func() { delete(mmprojUsage, "/proj") })

	pred := predictFromMeasurement(mem, launch, in)
	const fitMiB = 22144
	got := float64(pred.Total()) / mib
	// Above llama.cpp's requirement, by no more than the measure role's
	// larger compute buffer (589 vs 420 MiB) plus the draft compute bound.
	if got < fitMiB || got > fitMiB+400 {
		t.Errorf("required %.0f MiB, llama.cpp fit needs %d MiB", got, fitMiB)
	}
	if !pred.Measured || len(pred.ByDevice) != 1 {
		t.Errorf("prediction = %+v", pred)
	}

	// Before a runner has logged the projector's usage: size + headroom.
	delete(mmprojUsage, "/proj")
	learnedMiB := 1161.02
	learned := uint64(learnedMiB * mib)
	if unlearned := predictFromMeasurement(mem, launch, in); unlearned.Total()-pred.Total() != (888+1024)*mib-learned {
		t.Errorf("unlearned projector bound: %d vs %d", unlearned.Total(), pred.Total())
	}
}

func TestMMProjUsageLearnedFromLog(t *testing.T) {
	r := &llamaServerRunner{
		vramByDevice:     map[string]uint64{},
		systemFreeAtLoad: map[string]uint64{},
		launch:           llamaServerLaunchConfig{projectors: []string{"/learn"}},
	}
	t.Cleanup(func() { delete(mmprojUsage, "/learn") })
	w := &memoryParsingWriter{inner: io.Discard, runner: r}
	_, _ = w.Write([]byte("srv    load_model: [mtmd] estimated worst-case memory usage of mmproj is 1296.85 MiB (took 19.88 ms)\n"))
	approxMiB(t, "learned", mmprojUsage["/learn"], 1296.85)
}

// TestMeasureLlamaServerProtocol runs a fake llama-server to check the
// child environment and arguments.
func TestMeasureLlamaServerProtocol(t *testing.T) {
	dir := t.TempDir()
	argsFile := filepath.Join(dir, "args")
	script := "#!/bin/sh\n" +
		"echo \"$LLAMA_SERVER_ROUTER_PORT|$LLAMA_SERVER_CHILD_ROLE|$CUDA_VISIBLE_DEVICES|$*\" > " + argsFile + "\n" +
		"echo 'not the state line'\n" +
		"echo '" + qwen64kMeasureLine + "'\n"
	exe := filepath.Join(dir, "llama-server")
	if err := os.WriteFile(exe, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	orig := findMeasureServer
	findMeasureServer = func() (string, error) { return exe, nil }
	t.Cleanup(func() { findMeasureServer = orig })

	path, f := writeTestGGUF(t, qwen35KV(), nil)
	opts := api.DefaultOptions()
	opts.NumCtx = 8192
	gpu := cudaGPU("GPU-x", "CUDA1", 23962*mib)
	in := VRAMPredictInput{GPUs: []ml.DeviceInfo{gpu}, ModelPath: path, Model: f, Options: opts, NumParallel: 1}
	mem, err := MeasureLlamaServer(t.Context(), in)
	if err != nil {
		t.Fatal(err)
	}
	approxMiB(t, "model", mem.Devices[0].Model, 15339.44)

	b, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatal(err)
	}
	parts := strings.SplitN(strings.TrimSpace(string(b)), "|", 4)
	if parts[0] != "0" || parts[1] != "measure" || parts[2] != "GPU-x" {
		t.Errorf("child env = %q", parts[:3])
	}
	if !strings.Contains(parts[3], "-ngl all --fit off") || !strings.Contains(parts[3], "-c 8192") {
		t.Errorf("args = %q", parts[3])
	}

	// Cached: the fake binary is not run again.
	os.Remove(argsFile)
	if _, err := MeasureLlamaServer(t.Context(), in); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(argsFile); err == nil {
		t.Error("measurement not cached")
	}
}

func TestMMProjUsagePersisted(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	exe := filepath.Join(t.TempDir(), "llama-server")
	if err := os.WriteFile(exe, []byte("v1"), 0o755); err != nil {
		t.Fatal(err)
	}
	orig := findMeasureServer
	findMeasureServer = func() (string, error) { return exe, nil }
	reset := func() {
		mmprojUsageMu.Lock()
		clear(mmprojUsage)
		mmprojUsageLoaded = false
		mmprojUsageMu.Unlock()
	}
	t.Cleanup(func() { findMeasureServer = orig; reset() })
	reset()

	recordMMProjUsage("/blobs/proj", 1605.95)
	reset() // restart

	launch := llamaServerLaunchConfig{
		projectors: []string{"/blobs/proj"}, mmprojMemory: 1335 * mib, opts: api.DefaultOptions(),
		gpus: []ml.DeviceInfo{cudaGPU("GPU-1", "CUDA0", 23911*mib)},
	}
	approxMiB(t, "after restart", launch.mmprojFitUsage(), 1605.95)

	// A different llama-server build: back to the bound.
	if err := os.WriteFile(exe, []byte("v2 longer"), 0o755); err != nil {
		t.Fatal(err)
	}
	reset()
	if got := launch.mmprojFitUsage(); got != (1335<<20)+mmprojOffloadHeadroom {
		t.Errorf("new build kept usage %d", got)
	}
}

func TestMeasureKeyFreeMemory(t *testing.T) {
	args := []string{"--model", "m"}
	one := []ml.DeviceInfo{cudaGPU("a", "CUDA0", 20<<30)}
	oneLater := []ml.DeviceInfo{cudaGPU("a", "CUDA0", 5<<30)}
	if measureKey(args, nil, nil, one) != measureKey(args, nil, nil, oneLater) {
		t.Error("single-device key depends on free memory")
	}
	two := []ml.DeviceInfo{cudaGPU("a", "CUDA0", 20<<30), cudaGPU("b", "CUDA1", 20<<30)}
	twoLater := []ml.DeviceInfo{cudaGPU("a", "CUDA0", 3<<30), cudaGPU("b", "CUDA1", 20<<30)}
	if measureKey(args, nil, nil, two) == measureKey(args, nil, nil, twoLater) {
		t.Error("split key ignores free memory, which sets the split")
	}
}

func TestMeasureFailureRetried(t *testing.T) {
	dir := t.TempDir()
	exe := filepath.Join(dir, "llama-server")
	write := func(body string) {
		if err := os.WriteFile(exe, []byte("#!/bin/sh\n"+body+"\n"), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	orig := findMeasureServer
	findMeasureServer = func() (string, error) { return exe, nil }
	t.Cleanup(func() { findMeasureServer = orig })

	path, f := writeTestGGUF(t, qwen35KV(), nil)
	opts := api.DefaultOptions()
	opts.NumCtx = 4096 + 512 // unique cache key for this test
	in := VRAMPredictInput{GPUs: []ml.DeviceInfo{cudaGPU("GPU-r", "CUDA0", 23962*mib)}, ModelPath: path, Model: f, Options: opts, NumParallel: 1}

	write("echo 'cuda init failed' >&2; exit 1")
	if _, err := MeasureLlamaServer(t.Context(), in); err == nil {
		t.Fatal("failure not reported")
	}
	write("echo '" + qwen64kMeasureLine + "'")
	if _, err := MeasureLlamaServer(t.Context(), in); err == nil {
		t.Fatal("failure should be cached briefly")
	}
	// Expire the cached failure.
	measureCache.Lock()
	for k, r := range measureCache.m {
		if r.err != nil {
			r.expiry = time.Now().Add(-time.Second)
			measureCache.m[k] = r
		}
	}
	measureCache.Unlock()
	if _, err := MeasureLlamaServer(t.Context(), in); err != nil {
		t.Fatalf("not retried after expiry: %v", err)
	}
}

func TestMeasuredLowerBound(t *testing.T) {
	t.Setenv("OLLAMA_FLASH_ATTENTION", "1")
	p, err := parseMeasureOutput([]byte(qwen64kMeasureLine))
	if err != nil {
		t.Fatal(err)
	}
	gpu := cudaGPU("GPU-1", "CUDA0", 23911*mib)
	mem, err := memoryFromPayload(p, []ml.DeviceInfo{gpu}, nil)
	if err != nil {
		t.Fatal(err)
	}
	path, f := writeTestGGUF(t, qwen35KV(), nil)
	opts := api.DefaultOptions()
	opts.NumCtx = 65536
	opts.DraftNumPredict = 4
	in := VRAMPredictInput{GPUs: []ml.DeviceInfo{gpu}, ModelPath: path, Model: f, Options: opts, NumParallel: 1}
	launch := llamaServerLaunchConfig{
		modelPath: path, draftType: draftTypeMTP, projectors: []string{"/proj"}, mmprojMemory: 888 * mib,
		opts: opts, numParallel: 1, gpus: in.GPUs,
	}
	pred := predictFromMeasurement(mem, launch, in)
	// Logged: weights 15339.44, KV 2176 + RS 748.12, draft KV 256, projector 888.
	approxMiB(t, "lower bound", pred.MinTotal(), 15339.44+2176+748.125+256+888)
	if pred.MinTotal() >= pred.Total() {
		t.Errorf("lower bound %d not below requirement %d", pred.MinTotal(), pred.Total())
	}
}
