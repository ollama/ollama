package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/ml"
)

// llama-server's measure role (the one its router uses before loading a
// model): it loads the model and context with no_alloc, prints the memory
// breakdown per device as one JSON line on stdout and exits. Nothing is
// allocated. See tools/server/server.cpp server_run_measure.
const (
	llamaServerRouterPortEnv = "LLAMA_SERVER_ROUTER_PORT" // presence makes llama-server a router child
	llamaServerChildRoleEnv  = "LLAMA_SERVER_CHILD_ROLE"
	llamaServerMeasureRole   = "measure"
	llamaServerStatePrefix   = "cmd_child_to_router:state:"
	llamaArgFitEnv           = "LLAMA_ARG_FIT"

	measureTimeout = 2 * time.Minute
	// Cap on the new process's own context cost (CUDA context etc.) taken
	// from a measurement; a larger gap means free memory moved meanwhile.
	maxProcessOverhead = 512 << 20
)

// DeviceMemory is llama-server's measured memory for one GPU.
type DeviceMemory struct {
	ID                      ml.DeviceID
	Model, Context, Compute uint64
	// Overhead is the free memory a new llama-server process costs before
	// allocating anything (its CUDA context), as seen at measurement time.
	Overhead uint64
}

// Self is the device memory llama-server allocates for the main context.
func (d DeviceMemory) Self() uint64 { return d.Model + d.Context + d.Compute }

// LlamaServerMemory is llama-server's measurement of a launch. It covers the
// main model and context only: not the draft/MTP context or the projector.
type LlamaServerMemory struct {
	Devices   []DeviceMemory // in the order of the measured gpus
	HostModel uint64         // model weights in host buffers
}

type measureDevice struct {
	Kind    string `json:"kind"`
	Name    string `json:"name"`
	Model   uint64 `json:"model"`
	Context uint64 `json:"context"`
	Compute uint64 `json:"compute"`
	Total   uint64 `json:"total"`
	Free    uint64 `json:"free"`
}

type measurePayload struct {
	Result string `json:"result"`
	Error  string `json:"error"`
	Max    struct {
		NCtx    uint32          `json:"n_ctx"`
		Devices []measureDevice `json:"devices"`
	} `json:"max"`
}

// parseMeasureOutput extracts the measure child's payload from its stdout.
func parseMeasureOutput(stdout []byte) (measurePayload, error) {
	var payload measurePayload
	for line := range bytes.Lines(stdout) {
		rest, ok := bytes.CutPrefix(bytes.TrimSpace(line), []byte(llamaServerStatePrefix))
		if !ok {
			continue
		}
		var msg struct {
			Payload measurePayload `json:"payload"`
		}
		if err := json.Unmarshal(rest, &msg); err != nil {
			return payload, fmt.Errorf("parse measurement: %w", err)
		}
		payload = msg.Payload
		if payload.Result != "measured" {
			if payload.Error == "" {
				payload.Error = payload.Result
			}
			return payload, fmt.Errorf("llama-server measurement failed: %s", payload.Error)
		}
		return payload, nil
	}
	return payload, errors.New("llama-server printed no measurement")
}

// memoryFromPayload maps the measured devices back to gpus. names maps each
// GPU to the name llama-server logs it under.
func memoryFromPayload(p measurePayload, gpus []ml.DeviceInfo, names map[ml.DeviceID]string) (LlamaServerMemory, error) {
	var mem LlamaServerMemory
	byName := map[string]measureDevice{}
	for _, d := range p.Max.Devices {
		switch d.Kind {
		case "device":
			byName[d.Name] = d
		case "host":
			mem.HostModel += d.Model
		}
	}
	used := map[string]bool{}
	for _, gpu := range gpus {
		name := names[gpu.DeviceID]
		if name == "" {
			name = gpu.Name
		}
		d, ok := byName[name]
		if !ok {
			return LlamaServerMemory{}, fmt.Errorf("measurement has no device %s", name)
		}
		used[name] = true
		dm := DeviceMemory{ID: gpu.DeviceID, Model: d.Model, Context: d.Context, Compute: d.Compute}
		if gpu.FreeMemory > d.Free {
			dm.Overhead = min(gpu.FreeMemory-d.Free, maxProcessOverhead)
		}
		mem.Devices = append(mem.Devices, dm)
	}
	// A device we did not ask for holding part of the model can't be
	// accounted for by the scheduler.
	for name, d := range byName {
		if !used[name] && d.Model+d.Context+d.Compute > 0 {
			return LlamaServerMemory{}, fmt.Errorf("measurement placed memory on unexpected device %s", name)
		}
	}
	return mem, nil
}

// measureArgs turns launch args into a full-offload measurement: every layer
// on GPU unless the user pinned a layer count, and --fit off so the result
// is the requirement rather than what fits right now.
func measureArgs(args []string, numGPU int) []string {
	out := make([]string, 0, len(args)+4)
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-ngl", "--fit":
			i++ // drop flag and value
			continue
		}
		out = append(out, args[i])
	}
	if numGPU >= 0 {
		out = append(out, "-ngl", strconv.Itoa(numGPU))
	} else {
		out = append(out, "-ngl", "all")
	}
	return append(out, "--fit", "off")
}

type measureResult struct {
	mem    LlamaServerMemory
	err    error
	expiry time.Time // failures are retried after a while
}

const measureErrorTTL = 30 * time.Second

var measureCache = struct {
	sync.Mutex
	m map[string]measureResult
}{m: map[string]measureResult{}}

const measureCacheSize = 256

// findMeasureServer locates llama-server; tests replace it.
var findMeasureServer = FindLlamaServer

// MeasureLlamaServer asks llama-server how much memory the launch the
// runner would make for these inputs needs on each of gpus with every layer
// offloaded. Results are cached per launch.
func MeasureLlamaServer(ctx context.Context, in VRAMPredictInput) (LlamaServerMemory, error) {
	mem, _, err := measureLaunch(ctx, in)
	return mem, err
}

// measureLaunch is MeasureLlamaServer, also returning the launch measured.
func measureLaunch(ctx context.Context, in VRAMPredictInput) (LlamaServerMemory, llamaServerLaunchConfig, error) {
	if len(in.GPUs) == 0 {
		return LlamaServerMemory{}, llamaServerLaunchConfig{}, errors.New("no GPUs to measure")
	}
	exe, err := findMeasureServer()
	if err != nil {
		return LlamaServerMemory{}, llamaServerLaunchConfig{}, err
	}
	launch, split, deviceEnvs, err := in.launch()
	if err != nil {
		return LlamaServerMemory{}, launch, err
	}
	defer func() { _, _ = removeSplitModelDirs(split.dirs) }()

	args := measureArgs(llamaServerArgs(launch, 0), in.Options.NumGPU)
	key := measureKey(args, split.dirs, deviceEnvs, in.GPUs)

	measureCache.Lock()
	if r, ok := measureCache.m[key]; ok && (r.err == nil || time.Now().Before(r.expiry)) {
		measureCache.Unlock()
		return r.mem, launch, r.err
	}
	measureCache.Unlock()

	mem, err := runMeasure(ctx, exe, args, launch.gpuLibs, launch.extraEnvsForStart(), in.GPUs, runnerDeviceLogNames(in.GPUs, deviceEnvs))
	if ctx.Err() != nil {
		return mem, launch, err // cancelled: don't cache
	}
	measureCache.Lock()
	if len(measureCache.m) >= measureCacheSize {
		clear(measureCache.m)
	}
	measureCache.m[key] = measureResult{mem: mem, err: err, expiry: time.Now().Add(measureErrorTTL)}
	measureCache.Unlock()
	return mem, launch, err
}

// measureKey identifies a launch. Temporary split-model dirs differ per
// call, so they are blanked out. With --fit off, llama.cpp splits a model
// across devices in proportion to their free memory, so a multi-device key
// includes it (to 256 MiB).
func measureKey(args, tempDirs []string, deviceEnvs map[string]string, gpus []ml.DeviceInfo) string {
	var b strings.Builder
	for _, a := range args {
		for _, dir := range tempDirs {
			a = strings.ReplaceAll(a, dir, "")
		}
		b.WriteString(a)
		b.WriteByte(0)
	}
	keys := make([]string, 0, len(deviceEnvs))
	for k := range deviceEnvs {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, k := range keys {
		fmt.Fprintf(&b, "%s=%s\x00", k, deviceEnvs[k])
	}
	for _, g := range gpus {
		fmt.Fprintf(&b, "%s/%s\x00", g.Library, g.ID)
		if len(gpus) > 1 {
			fmt.Fprintf(&b, "free=%d\x00", g.FreeMemory>>28)
		}
	}
	if v, ok := os.LookupEnv(llamaArgFitTargetEnv); ok {
		fmt.Fprintf(&b, "fit-target=%s", v)
	}
	return b.String()
}

func runMeasure(ctx context.Context, exe string, args, gpuLibs []string, env map[string]string, gpus []ml.DeviceInfo, names map[ml.DeviceID]string) (LlamaServerMemory, error) {
	ctx, cancel := context.WithTimeout(ctx, measureTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, exe, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &tailWriter{buf: &stderr, max: 4096}
	cmd.SysProcAttr = LlamaServerSysProcAttr
	envs := cloneStringMap(env)
	envs[llamaServerRouterPortEnv] = "0"
	envs[llamaServerChildRoleEnv] = llamaServerMeasureRole
	SetupLlamaServerCommandEnv(cmd, exe, gpuLibs, envs)

	start := time.Now()
	runErr := cmd.Run()
	payload, err := parseMeasureOutput(stdout.Bytes())
	if err != nil {
		if runErr != nil {
			err = fmt.Errorf("%w (%v): %s", err, runErr, strings.TrimSpace(stderr.String()))
		}
		return LlamaServerMemory{}, err
	}
	mem, err := memoryFromPayload(payload, gpus, names)
	if err != nil {
		return LlamaServerMemory{}, err
	}
	slog.Debug("measured llama-server memory", "took", time.Since(start), "n_ctx", payload.Max.NCtx, "devices", mem.Devices)
	return mem, nil
}

// tailWriter keeps the last max bytes written.
type tailWriter struct {
	buf *bytes.Buffer
	max int
}

func (w *tailWriter) Write(p []byte) (int, error) {
	w.buf.Write(p)
	if extra := w.buf.Len() - w.max; extra > 0 {
		w.buf.Next(extra)
	}
	return len(p), nil
}

// mmproj worst-case usage, learned from llama-server's log per projector
// and kept across restarts in the user cache dir. It does not depend on the
// context or batch size; a different llama-server binary discards it.
var (
	mmprojUsageMu     sync.Mutex
	mmprojUsage       = map[string]uint64{}
	mmprojUsageLoaded bool

	mmprojUsageRegex = regexp.MustCompile(`estimated worst-case memory usage of mmproj is ([0-9.]+) MiB`)
)

type mmprojUsageCache struct {
	Server string            `json:"server"` // llama-server path and mtime
	Usage  map[string]uint64 `json:"usage"`  // projector path -> bytes
}

func mmprojUsagePath() string {
	dir, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "ollama", "mmproj-usage.json")
}

func llamaServerIdentity() string {
	exe, err := findMeasureServer()
	if err != nil {
		return ""
	}
	// The binary is a thin wrapper; the code is in the libraries beside it.
	entries, err := os.ReadDir(filepath.Dir(exe))
	if err != nil {
		return ""
	}
	var newest int64
	var size int64
	for _, e := range entries {
		if info, err := e.Info(); err == nil && info.Mode().IsRegular() {
			newest = max(newest, info.ModTime().UnixNano())
			size += info.Size()
		}
	}
	return fmt.Sprintf("%s:%d:%d", exe, size, newest)
}

// loadMMProjUsageLocked reads the persisted usage once. Requires mmprojUsageMu.
func loadMMProjUsageLocked() {
	if mmprojUsageLoaded {
		return
	}
	mmprojUsageLoaded = true
	path := mmprojUsagePath()
	if path == "" {
		return
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var c mmprojUsageCache
	if json.Unmarshal(b, &c) != nil || c.Server == "" || c.Server != llamaServerIdentity() {
		return
	}
	for k, v := range c.Usage {
		if _, ok := mmprojUsage[k]; !ok {
			mmprojUsage[k] = v
		}
	}
}

func recordMMProjUsage(projector string, mib float64) {
	if projector == "" || mib <= 0 {
		return
	}
	usage := uint64(mib * (1 << 20))
	mmprojUsageMu.Lock()
	defer mmprojUsageMu.Unlock()
	loadMMProjUsageLocked()
	if mmprojUsage[projector] == usage {
		return
	}
	mmprojUsage[projector] = usage

	path := mmprojUsagePath()
	server := llamaServerIdentity()
	if path == "" || server == "" {
		return
	}
	b, err := json.Marshal(mmprojUsageCache{Server: server, Usage: mmprojUsage})
	if err != nil {
		return
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		slog.Debug("cannot save mmproj usage", "error", err)
		return
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err == nil {
		err = os.Rename(tmp, path)
		if err != nil {
			slog.Debug("cannot save mmproj usage", "error", err)
		}
	}
}

// mmprojFitUsage is the projector memory llama-server adds to its fit target
// on the projector's device: the learned worst case, or until a runner has
// logged it, the projector size plus Ollama's offload headroom.
func (launch llamaServerLaunchConfig) mmprojFitUsage() uint64 {
	if len(launch.projectors) == 0 || launch.mmprojMemory == 0 {
		return 0
	}
	if disabled, _ := launch.mmprojOffloadDisabled(); disabled {
		return 0
	}
	mmprojUsageMu.Lock()
	loadMMProjUsageLocked()
	usage, ok := mmprojUsage[launch.projectors[0]]
	mmprojUsageMu.Unlock()
	if ok {
		return usage
	}
	return launch.mmprojMemory + mmprojOffloadHeadroom
}

// fitTargets is the free memory llama-server's --fit keeps on each of n
// devices before the projector is added: LLAMA_ARG_FIT_TARGET as the runner
// sets it (MiB, one value for all devices or one per device) or llama.cpp's
// 1024 MiB default. nil when --fit is off.
func (launch llamaServerLaunchConfig) fitTargets(n int) []uint64 {
	lookup := func(key string) (string, bool) {
		if v, ok := launch.extraEnvsForStart()[key]; ok {
			return v, true
		}
		return os.LookupEnv(key)
	}
	if v, ok := lookup(llamaArgFitEnv); ok {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "off", "0", "false":
			return nil
		}
	}
	targets := make([]uint64, n)
	for i := range targets {
		targets[i] = predictFitTargetDefault
	}
	v, ok := lookup(llamaArgFitTargetEnv)
	if !ok {
		return targets
	}
	// llama.cpp: one value is broadcast; a list sets devices in order.
	parts := fitTargetSplit.Split(strings.TrimSpace(v), -1)
	for i := range targets {
		p := parts[0]
		if len(parts) > 1 {
			if i >= len(parts) {
				break
			}
			p = parts[i]
		}
		if mib, err := strconv.ParseUint(strings.TrimSpace(p), 10, 64); err == nil {
			targets[i] = mib << 20
		}
	}
	return targets
}

var fitTargetSplit = regexp.MustCompile(`[,/]+`)

// VRAMPredictInput is a llama-server launch to predict, with the same inputs
// as NewLlamaServer.
type VRAMPredictInput struct {
	GPUs        []ml.DeviceInfo // the devices the model would use
	ModelPath   string
	Model       *gguf.Model
	Adapters    []string
	Projectors  []string
	Options     api.Options // NumCtx per sequence
	NumParallel int
	Config      LlamaServerConfig
}

func (in VRAMPredictInput) launch() (llamaServerLaunchConfig, splitModelLaunch, map[string]string, error) {
	opts := in.Options
	opts.NumCtx = clampTrainContext(opts.NumCtx, in.Model)
	return newLlamaServerLaunch(in.GPUs, in.ModelPath, in.Model, in.Adapters, in.Projectors, opts,
		max(in.NumParallel, 1), strings.ToLower(envconfig.KvCacheType()), in.Config)
}

// clampTrainContext caps numCtx at the model's training context, as
// NewLlamaServer does.
func clampTrainContext(numCtx int, f *gguf.Model) int {
	if f == nil {
		return numCtx
	}
	if train := int(f.KV().ContextLength()); train > 0 && numCtx > train {
		return train
	}
	return numCtx
}

// EstimateLlamaServer predicts from GGUF metadata alone (vram_predict.go).
func EstimateLlamaServer(in VRAMPredictInput) VRAMPrediction {
	ctx := clampTrainContext(in.Options.NumCtx, in.Model) * max(in.NumParallel, 1)
	return PredictServerVRAM(in.ModelPath, in.Model, in.Projectors, in.Options, VRAMPredictConfig{
		NumCtx:         ctx,
		NumParallel:    in.NumParallel,
		Devices:        len(in.GPUs),
		FlashAttention: LlamaServerFlashAttention(in.GPUs),
		DraftModelPath: in.Config.DraftModelPath,
	})
}

var measureFallbackLogged sync.Map

// PredictLlamaServer predicts the memory a launch needs on each of in.GPUs
// for llama-server to keep every layer on GPU. It mirrors the check
// llama.cpp's --fit makes at startup, per device:
//
//	measured main context + draft context + fit target (+ projector) <= free
//
// plus the new process's own overhead, so the result compares against the
// scheduler's free memory. The main context is llama-server's own
// measurement; the draft context and projector, which the measurement
// leaves out, are estimated. Falls back to EstimateLlamaServer when
// llama-server can't measure.
func PredictLlamaServer(ctx context.Context, in VRAMPredictInput) VRAMPrediction {
	mem, launch, err := measureLaunch(ctx, in)
	if err != nil {
		key := in.ModelPath + "\x00" + err.Error()
		if _, logged := measureFallbackLogged.LoadOrStore(key, true); !logged {
			slog.Warn("llama-server memory measurement failed; using metadata estimate", "model", in.ModelPath, "error", err)
		}
		return EstimateLlamaServer(in)
	}
	return predictFromMeasurement(mem, launch, in)
}

func predictFromMeasurement(mem LlamaServerMemory, launch llamaServerLaunchConfig, in VRAMPredictInput) VRAMPrediction {
	draftKV, draftCompute := draftContext(launch, in)
	targets := launch.fitTargets(len(mem.Devices))
	mmproj := launch.mmprojFitUsage()
	var mmprojWeights uint64
	if mmproj > 0 {
		mmprojWeights = launch.mmprojMemory
	}

	pred := VRAMPrediction{Devices: len(mem.Devices), Measured: true}
	for i, d := range mem.Devices {
		need := d.Self() + d.Overhead + draftKV + draftCompute
		if targets != nil {
			// llama-server adds the projector to the target of the device
			// it runs on; which one that is in a split isn't known here.
			need += targets[i] + mmproj
		}
		pred.ByDevice = append(pred.ByDevice, need)

		least := d.Model + d.Context + draftKV
		if i == 0 {
			least += mmprojWeights // on one device, which isn't known
		}
		pred.MinByDevice = append(pred.MinByDevice, least)
	}
	return pred
}

// draftContext estimates the speculative draft context llama-server creates
// next to the main one (not in the measurement): its KV cache (exact for the
// MTP head: f16, nextn layers, full context) and compute buffer (an upper
// bound), or an external draft model's with an f16 cache. Counted on every
// device, since its placement isn't known.
func draftContext(launch llamaServerLaunchConfig, in VRAMPredictInput) (kv, compute uint64) {
	if launch.opts.DraftNumPredict <= 0 {
		return 0, 0
	}
	p := vramPredictParams{
		numCtx:      launch.opts.NumCtx * max(launch.numParallel, 1),
		numParallel: launch.numParallel,
		numUBatch:   launch.opts.NumBatch,
		cacheType:   "f16",
		flashAttn:   LlamaServerFlashAttention(in.GPUs) != ml.FlashAttentionDisabled,
		devices:     len(in.GPUs),
	}
	switch {
	case launch.config.DraftModelPath != "":
		df, err := LoadModel(launch.config.DraftModelPath, 1024)
		if err != nil {
			return 0, 0
		}
		d := newVRAMLayout(launch.config.DraftModelPath, df, nil).breakdown(p)
		return d.weights + d.kv + d.rs, d.compute
	case launch.draftType == draftTypeMTP:
		p.draftN = launch.opts.DraftNumPredict
		d := newVRAMLayout(in.ModelPath, in.Model, nil).breakdown(p)
		return d.draftKV, d.draftCompute
	}
	return 0, 0
}
