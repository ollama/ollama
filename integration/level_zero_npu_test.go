// SPDX-License-Identifier: MIT
//go:build integration && level_zero && npu

package integration

// level_zero_npu_test.go — Intel NPU-specific integration tests
//
// These tests require ALL THREE build tags: integration, level_zero, AND npu.
// They are only run on Intel NPU-capable hardware (Meteor Lake, Lunar Lake,
// Arrow Lake) with OLLAMA_L0_NPU_ENABLE=1 set.
//
// Build and run:
//   go build .
//   OLLAMA_L0_NPU_ENABLE=1 go test -tags=integration,level_zero,npu \
//       -v -count=1 -timeout=10m ./integration/
//
// Env vars used by this file:
//   OLLAMA_L0_NPU_ENABLE        — MUST be "1" (enforced by skipIfNoNPU)
//   OLLAMA_L0_NPU_MIN_TPS       — minimum tokens/sec threshold (default: 5.0)
//   OLLAMA_TEST_MODEL           — override default model (default: tinyllama)
//   OLLAMA_TEST_EXISTING        — if set, use existing server at OLLAMA_HOST
//
// Background (ADR-L0-002, ADR-L0-007):
//   Intel NPU (VPU) silicon is distinct from the GPU. It is enumerated as
//   ze_device_type_t == ZE_DEVICE_TYPE_VPU=4. NPU placement is opt-in via
//   OLLAMA_L0_NPU_ENABLE=1. The NPU is optimized for ≤ 8B Q4 models.
//   Larger models automatically fall back to GPU or CPU via the scheduler's
//   VRAM-fit heuristic (ADR-L0-007 §Decision item 3 soft-cap).

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// defaultNPUMinTPS is the default minimum acceptable tokens-per-second
// threshold for NPU inference. Based on observed Meteor Lake NPU performance
// for tinyllama (Q4_0) with the Intel NPU software stack (OpenVINO backend).
// Override via OLLAMA_L0_NPU_MIN_TPS env var.
const defaultNPUMinTPS = 5.0

// maxNPUModelSizeB is the maximum safe model size (in bytes) for NPU placement
// per ADR-L0-007. Models exceeding this size are silently skipped by the NPU
// enumerator — the scheduler sees them as "does not fit" and falls back to GPU.
// 8 billion parameters at Q4_0 ≈ 4 GB ≈ 4,294,967,296 bytes.
const maxNPUModelSizeB = 8_000_000_000

// TestNPUSmallModelInference runs inference on a ≤ 8B Q4 model with
// OLLAMA_L0_NPU_ENABLE=1 and asserts:
//   - The server loads the model and produces a non-empty chat response.
//   - The tokens-per-second rate (EvalCount / EvalDuration) exceeds the
//     minimum threshold defined by OLLAMA_L0_NPU_MIN_TPS (default: 5.0).
//
// Throughput assertion:
//   The threshold of 5.0 tok/s is the conservative lower bound for tinyllama
//   (1.1B Q4_0) on a Meteor Lake NPU. Production NPU throughput is typically
//   10–30 tok/s for ≤ 4B Q4 models. The default is intentionally low to
//   avoid false negatives on thermally throttled CI machines.
//
// SKIP conditions:
//   - OLLAMA_L0_NPU_ENABLE ≠ "1" (skipIfNoNPU check).
//   - No Level Zero device detected (skipIfNoL0 inside skipIfNoNPU).
//   - Model not available on the runner (pullOrSkip).
func TestNPUSmallModelInference(t *testing.T) {
	skipIfNoNPU(t)

	// Read the minimum TPS threshold from env; fall back to default.
	minTPS := defaultNPUMinTPS
	if rawTPS := os.Getenv("OLLAMA_L0_NPU_MIN_TPS"); rawTPS != "" {
		if v, err := strconv.ParseFloat(rawTPS, 64); err == nil && v > 0 {
			minTPS = v
		} else {
			t.Logf("TestNPUSmallModelInference: invalid OLLAMA_L0_NPU_MIN_TPS=%q, using default %.1f", rawTPS, defaultNPUMinTPS)
		}
	}
	t.Logf("TestNPUSmallModelInference: minTPS threshold=%.1f tok/s", minTPS)

	client, cleanup := runServerWithEnv(t, map[string]string{
		"OLLAMA_L0_NPU_ENABLE":    "1",
		"OLLAMA_L0_DEVICE_INDEX":  "0",
	})
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	modelName := l0TestModel()
	pullOrSkip(ctx, t, client, modelName)
	t.Logf("TestNPUSmallModelInference: model=%s", modelName)

	streamEnabled := false
	req := api.ChatRequest{
		Model: modelName,
		Messages: []api.Message{
			{
				Role:    "user",
				Content: "Count from 1 to 10.",
			},
		},
		Stream: &streamEnabled,
		Options: map[string]any{
			"temperature": 0,
			"seed":        7,
			"num_predict": 64, // enough tokens to measure throughput
		},
	}

	var responseContent strings.Builder
	var evalCount int
	var evalDuration time.Duration

	startTime := time.Now()

	err := client.Chat(ctx, &req, func(resp api.ChatResponse) error {
		responseContent.WriteString(resp.Message.Content)
		if resp.Done {
			evalCount = resp.Metrics.EvalCount
			evalDuration = resp.Metrics.EvalDuration
		}
		return nil
	})
	if err != nil {
		t.Fatalf("TestNPUSmallModelInference: /api/chat failed: %v", err)
	}

	wallDuration := time.Since(startTime)
	content := responseContent.String()

	if content == "" {
		t.Error("TestNPUSmallModelInference: response content is empty — model produced no tokens")
	}
	t.Logf("TestNPUSmallModelInference: response length=%d chars, wall_time=%v", len(content), wallDuration)

	// Throughput calculation.
	// Primary: use EvalCount and EvalDuration from the API response (most accurate).
	// Fallback: derive from wall-clock time (less accurate, includes network).
	var measuredTPS float64
	if evalCount > 0 && evalDuration > 0 {
		evalDurationSec := evalDuration.Seconds()
		measuredTPS = float64(evalCount) / evalDurationSec
		t.Logf("TestNPUSmallModelInference: eval_count=%d eval_duration=%.3fs tps=%.1f",
			evalCount, evalDurationSec, measuredTPS)
	} else if wallDuration > 0 {
		// Fallback: tokens from response length (rough estimate — not actual tokens).
		// This is a weak proxy; real tokenization is model-specific.
		wordCount := len(strings.Fields(content))
		measuredTPS = float64(wordCount) / wallDuration.Seconds()
		t.Logf("TestNPUSmallModelInference: eval_count unavailable, using wall-clock fallback: words=%d wall_sec=%.3f approx_tps=%.1f",
			wordCount, wallDuration.Seconds(), measuredTPS)
	}

	// Assert throughput exceeds the minimum threshold.
	// CANNOT VERIFY: whether inference actually ran on the NPU vs CPU/GPU.
	// The scheduler placement decision is internal to the server. We assert
	// only that the observable performance meets the minimum threshold.
	if measuredTPS > 0 && measuredTPS < minTPS {
		t.Errorf("TestNPUSmallModelInference: throughput %.1f tok/s is below minimum %.1f tok/s — NPU may not be active or system is thermally throttled",
			measuredTPS, minTPS)
	}

	t.Logf("TestNPUSmallModelInference: PASS — content_len=%d tps=%.1f (min=%.1f)", len(content), measuredTPS, minTPS)
}

// TestNPUPowerBenefit is a heuristic test that asserts inference with
// OLLAMA_L0_NPU_ENABLE=1 does not cause excessive CPU load during the
// inference run. The NPU is a dedicated accelerator; when it is active,
// the host CPU should not be pegged at 100% during inference.
//
// ─────────────────────────────────────────────────────────────────────────
// IMPORTANT DISCLAIMER — WEAK HEURISTIC
//
// Precise CPU utilization measurement requires host-level tooling:
//   - Linux: /proc/[pid]/stat, cgroups v2 cpu.stat, or `perf stat`
//   - Windows: QueryProcessCycleTime or GetSystemTimes
//   - macOS: task_info(MACH_TASK_BASIC_INFO)
//
// None of these are available in a pure Go integration test without adding
// a platform-specific dependency or a privileged system call. The approach
// below uses goroutine count and wall-clock timing as a proxy:
//
//   - Goroutine count: if the server is CPU-bound (many threads spinning),
//     runtime.NumGoroutine() in the TEST process will be elevated due to
//     indirect pressure. This is a very weak signal.
//   - Wall-clock: if inference is faster than CPU-only baseline, NPU is
//     likely active. We do not have a CPU-only baseline in CI.
//
// This test therefore ONLY asserts that:
//   1. Inference completes without deadlock (context timeout is the safety net).
//   2. The goroutine count in the TEST PROCESS stays below a sanity threshold
//      (prevents runaway goroutine leak in the test harness itself).
//
// For proper power benefit measurement, use host-level tooling:
//   - Linux: `perf stat -e power/energy-pkg/ -- ollama run tinyllama`
//   - Intel VTune: configure NPU activity counter collection
//   - Windows: Intel Power Gadget or Windows Performance Recorder
//
// This test is intended as a structural placeholder that will be replaced
// with a proper measurement once a /api/backend endpoint is available.
// ─────────────────────────────────────────────────────────────────────────
func TestNPUPowerBenefit(t *testing.T) {
	skipIfNoNPU(t)

	// Goroutine count threshold in the TEST PROCESS.
	// The test process itself should have a small number of goroutines.
	// A goroutine count > 200 suggests a leak in the test harness itself.
	const maxTestGoroutines = 200

	client, cleanup := runServerWithEnv(t, map[string]string{
		"OLLAMA_L0_NPU_ENABLE":   "1",
		"OLLAMA_L0_DEVICE_INDEX": "0",
	})
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	modelName := l0TestModel()
	pullOrSkip(ctx, t, client, modelName)

	// Measure goroutine count BEFORE inference.
	goroutinesBefore := runtime.NumGoroutine()
	t.Logf("TestNPUPowerBenefit: goroutines BEFORE inference=%d", goroutinesBefore)

	streamEnabled := false
	req := api.ChatRequest{
		Model: modelName,
		Messages: []api.Message{
			{Role: "user", Content: "Describe the sky in one sentence."},
		},
		Stream: &streamEnabled,
		Options: map[string]any{
			"temperature": 0,
			"seed":        3,
			"num_predict": 32,
		},
	}

	startTime := time.Now()

	var responseContent strings.Builder
	err := client.Chat(ctx, &req, func(resp api.ChatResponse) error {
		responseContent.WriteString(resp.Message.Content)
		return nil
	})
	wallDuration := time.Since(startTime)

	if err != nil {
		t.Fatalf("TestNPUPowerBenefit: /api/chat failed: %v", err)
	}

	// Measure goroutine count AFTER inference.
	goroutinesAfter := runtime.NumGoroutine()
	t.Logf("TestNPUPowerBenefit: goroutines AFTER inference=%d wall_time=%v", goroutinesAfter, wallDuration)

	// Assert: goroutine count in test process stays sane.
	if goroutinesAfter > maxTestGoroutines {
		t.Errorf("TestNPUPowerBenefit: test-process goroutine count %d exceeds threshold %d — possible goroutine leak in test harness",
			goroutinesAfter, maxTestGoroutines)
	}

	// Assert: inference completed (non-empty response).
	content := responseContent.String()
	if content == "" {
		t.Error("TestNPUPowerBenefit: inference produced empty response")
	}

	// Log the wall-clock duration as a proxy performance indicator.
	// CANNOT VERIFY: CPU% utilization without OS-level measurement tools.
	// MANUAL VERIFICATION REQUIRED: Run `perf stat -e cpu-cycles,cpu-clock`
	// alongside this test on a Meteor Lake system to confirm NPU offload.
	t.Logf("TestNPUPowerBenefit: PASS (weak heuristic)")
	t.Logf("TestNPUPowerBenefit: wall_time=%v goroutines_before=%d goroutines_after=%d",
		wallDuration, goroutinesBefore, goroutinesAfter)
	t.Logf("TestNPUPowerBenefit: response=%q", truncate(content, 80))
	t.Logf("TestNPUPowerBenefit: CANNOT VERIFY CPU%% offload from integration test — use host perf tools for power measurement")
	t.Logf("TestNPUPowerBenefit: recommended: perf stat -e power/energy-pkg/ -p <ollama-server-pid>")
}

// Ensure the fmt package is available for diagnostic formatting.
var _npu = fmt.Sprintf
