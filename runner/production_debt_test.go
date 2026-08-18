package runner

import (
	"testing"
)

func TestCleanRunnerExecutionPassesReadiness(t *testing.T) {
	gate := NewProductionDebtRunnerGate(true, 12.0)

	report, err := gate.EvaluateRunnerExecution(
		"ollama_llama3_70b_metal_runner",
		80,
		80,
		40000000000,
		41000000000,
		0.45,
		0,
		0,
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !report.IsProductionReady {
		t.Errorf("expected is_production_ready to be true, got false")
	}
	if report.ODIScore > 12.0 {
		t.Errorf("expected odi_score <= 12.0, got %f", report.ODIScore)
	}
	if len(report.CriticalSmells) != 0 {
		t.Errorf("expected 0 critical smells, got %d", len(report.CriticalSmells))
	}
	if report.ReceiptHash == "" {
		t.Errorf("expected non-empty receipt_hash")
	}
}

func TestDegradedRunnerExecutionFailsDebt(t *testing.T) {
	gate := NewProductionDebtRunnerGate(true, 12.0)

	report, err := gate.EvaluateRunnerExecution(
		"uncalibrated_runner_instance",
		10,
		80, // Insufficient offload
		20000000000,
		55000000000, // 2.75x context VRAM sprawl
		4.5,         // High TTFT latency
		3,           // 3 model swapping failures
		2,           // 2 un-gated mutations
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if report.IsProductionReady {
		t.Errorf("expected is_production_ready to be false, got true")
	}
	if report.ODIScore < 50.0 {
		t.Errorf("expected odi_score > 50.0, got %f", report.ODIScore)
	}

	hasVramSprawl := false
	hasTTFT := false
	hasSwapping := false
	hasMutations := false
	for _, s := range report.CriticalSmells {
		if s == "HIGH_CONTEXT_VRAM_SPRAWL_2.75X" {
			hasVramSprawl = true
		}
		if s == "HIGH_TTFT_LATENCY_4.50S" {
			hasTTFT = true
		}
		if s == "DETECTED_3_MODEL_SWAPPING_FAILURES" {
			hasSwapping = true
		}
		if s == "DETECTED_2_UNGATED_LOCAL_MUTATIONS" {
			hasMutations = true
		}
	}

	if !hasVramSprawl || !hasTTFT || !hasSwapping || !hasMutations {
		t.Errorf("missing expected critical smells: %v", report.CriticalSmells)
	}
}

func TestCryptographicLedgerIntegrity(t *testing.T) {
	gate := NewProductionDebtRunnerGate(true, 12.0)

	_, _ = gate.EvaluateRunnerExecution("runner-1", 80, 80, 1000, 1000, 0.2, 0, 0)
	_, _ = gate.EvaluateRunnerExecution("runner-2", 80, 80, 1000, 1000, 0.2, 0, 0)
	_, _ = gate.EvaluateRunnerExecution("runner-3", 80, 80, 1000, 1000, 0.2, 0, 0)

	entries := gate.Ledger.GetLedgerEntries()
	if len(entries) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(entries))
	}
	if entries[0]["prev_hash"] != GenesisHash {
		t.Errorf("expected genesis hash for entry 0")
	}
	if entries[1]["prev_hash"] != entries[0]["curr_hash"] {
		t.Errorf("entry 1 prev_hash mismatch")
	}
	if entries[2]["prev_hash"] != entries[1]["curr_hash"] {
		t.Errorf("entry 2 prev_hash mismatch")
	}
	if !gate.Ledger.VerifyLedgerIntegrity() {
		t.Errorf("expected ledger integrity to be true")
	}
}
