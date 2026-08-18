package runner

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"
)

const GenesisHash = "0000000000000000000000000000000000000000000000000000000000000000"

// OllamaDebtReport contains quantitative evaluation of Ollama runner execution.
type OllamaDebtReport struct {
	RunnerID                  string   `json:"runner_id"`
	ODIScore                  float64  `json:"odi_score"`                   // Ollama Debt Index (target <= 12.0)
	ContextLayerSprawl        float64  `json:"context_layer_sprawl"`        // Target <= 1.08x
	TTFTLatencySeconds        float64  `json:"ttft_latency_seconds"`        // Target <= 0.65s
	MutationSafetyScore       float64  `json:"mutation_safety_score"`       // Target 100.0
	ProductionReadinessIndex  float64  `json:"production_readiness_index"`  // Scale 0 - 100
	IsProductionReady         bool     `json:"is_production_ready"`
	CriticalSmells            []string `json:"critical_smells"`
	ReceiptHash               string   `json:"receipt_hash"`
}

// TechnicalDueDiligenceLedger manages an immutable SHA-256 hash-chained Action Ledger for Ollama.
type TechnicalDueDiligenceLedger struct {
	mu       sync.Mutex
	entries  []map[string]any
	lastHash string
}

func NewTechnicalDueDiligenceLedger() *TechnicalDueDiligenceLedger {
	return &TechnicalDueDiligenceLedger{
		entries:  make([]map[string]any, 0),
		lastHash: GenesisHash,
	}
}

func (l *TechnicalDueDiligenceLedger) RecordRunnerEvent(
	runnerID string,
	eventType string,
	readinessIndex float64,
	criticalSmells []string,
	metadata map[string]any,
) map[string]any {
	l.mu.Lock()
	defer l.mu.Unlock()

	timestamp := time.Now().UTC().Format(time.RFC3339Nano)
	index := len(l.entries)

	metaBytes, _ := json.Marshal(metadata)
	metaHash := sha256.Sum256(metaBytes)
	metaHashHex := hex.EncodeToString(metaHash[:])

	canonicalContent := fmt.Sprintf("%d|%s|%s|%s|%.2f|%s|%s",
		index, l.lastHash, runnerID, eventType, readinessIndex, timestamp, metaHashHex)
	currHashArr := sha256.Sum256([]byte(canonicalContent))
	currHash := hex.EncodeToString(currHashArr[:])

	entry := map[string]any{
		"index":           index,
		"timestamp":       timestamp,
		"runner_id":       runnerID,
		"event_type":      eventType,
		"readiness_index": readinessIndex,
		"critical_smells": criticalSmells,
		"prev_hash":       l.lastHash,
		"curr_hash":       currHash,
		"metadata":        metadata,
	}

	l.entries = append(l.entries, entry)
	l.lastHash = currHash
	return entry
}

func (l *TechnicalDueDiligenceLedger) GetLedgerEntries() []map[string]any {
	l.mu.Lock()
	defer l.mu.Unlock()
	copied := make([]map[string]any, len(l.entries))
	copy(copied, l.entries)
	return copied
}

func (l *TechnicalDueDiligenceLedger) VerifyLedgerIntegrity() bool {
	l.mu.Lock()
	defer l.mu.Unlock()

	prev := GenesisHash
	for _, entry := range l.entries {
		if entry["prev_hash"] != prev {
			return false
		}
		prev = entry["curr_hash"].(string)
	}
	return true
}

// ProductionDebtRunnerGate enforces architectural KPIs across Ollama runner models.
type ProductionDebtRunnerGate struct {
	NeverEquateIntentToApproval bool
	MaxAcceptableODI            float64
	Ledger                      *TechnicalDueDiligenceLedger
}

func NewProductionDebtRunnerGate(neverEquateIntent bool, maxAcceptableODI float64) *ProductionDebtRunnerGate {
	return &ProductionDebtRunnerGate{
		NeverEquateIntentToApproval: neverEquateIntent,
		MaxAcceptableODI:            maxAcceptableODI,
		Ledger:                      NewTechnicalDueDiligenceLedger(),
	}
}

func (g *ProductionDebtRunnerGate) CheckKillSwitch() bool {
	if os.Getenv("AAG_KILL_SWITCH") == "true" || os.Getenv("AAG_KILL_SWITCH") == "1" {
		return true
	}
	for _, p := range []string{"artifacts/KILL", "/tmp/KILL"} {
		if _, err := os.Stat(filepath.Clean(p)); err == nil {
			return true
		}
	}
	return false
}

func (g *ProductionDebtRunnerGate) EvaluateRunnerExecution(
	runnerID string,
	allocatedGPULayers int,
	totalModelLayers int,
	allocatedVRAMBytes int64,
	peakContextVRAMBytes int64,
	ttftLatencySeconds float64,
	modelSwappingFailures int,
	unGatedMutations int,
) (*OllamaDebtReport, error) {
	if g.CheckKillSwitch() {
		g.Ledger.RecordRunnerEvent(
			runnerID,
			"runner_halted_kill_switch",
			0.0,
			[]string{"EMERGENCY_KILL_SWITCH_ENGAGED"},
			map[string]any{"reason": "AAG_KILL_SWITCH is set"},
		)
		return nil, fmt.Errorf("A2Z SOC ActionGate: Emergency kill switch is engaged. Ollama runner halted")
	}

	criticalSmells := make([]string, 0)

	// KPI 2: Context Layer Sprawl Multiplier
	vramRatio := float64(peakContextVRAMBytes) / math.Max(1.0, float64(allocatedVRAMBytes))
	if vramRatio > 1.8 {
		criticalSmells = append(criticalSmells, fmt.Sprintf("HIGH_CONTEXT_VRAM_SPRAWL_%.2fX", vramRatio))
	}

	// Layer Offloading Ratio
	layerRatio := float64(allocatedGPULayers) / math.Max(1.0, float64(totalModelLayers))
	if layerRatio < 0.5 && totalModelLayers > 10 {
		criticalSmells = append(criticalSmells, fmt.Sprintf("INSUFFICIENT_GPU_OFFLOAD_%.2f", layerRatio))
	}

	// KPI 3: Latency Ceiling
	if ttftLatencySeconds > 3.0 {
		criticalSmells = append(criticalSmells, fmt.Sprintf("HIGH_TTFT_LATENCY_%.2fS", ttftLatencySeconds))
	}

	// Model swapping failures
	if modelSwappingFailures > 1 {
		criticalSmells = append(criticalSmells, fmt.Sprintf("DETECTED_%d_MODEL_SWAPPING_FAILURES", modelSwappingFailures))
	}

	// KPI 4: Mutation Safety
	if unGatedMutations > 0 {
		criticalSmells = append(criticalSmells, fmt.Sprintf("DETECTED_%d_UNGATED_LOCAL_MUTATIONS", unGatedMutations))
	}

	// KPI 1: Ollama Debt Index (0 = Clean, 100 = Catastrophic)
	odi := math.Max(0.0, (vramRatio-1.0)*20.0) +
		math.Max(0.0, (ttftLatencySeconds-0.65)*10.0) +
		float64(modelSwappingFailures)*15.0 +
		float64(unGatedMutations)*30.0

	odiScore := math.Min(100.0, math.Round(odi*100)/100)
	readiness := math.Max(0.0, 100.0-odiScore)
	isProductionReady := odiScore <= g.MaxAcceptableODI && len(criticalSmells) == 0

	mutationScore := 100.0
	if unGatedMutations > 0 {
		mutationScore = math.Max(0.0, 100.0-float64(unGatedMutations)*30.0)
	}

	metadata := map[string]any{
		"odi_score":                       odiScore,
		"vram_ratio":                      vramRatio,
		"layer_ratio":                     layerRatio,
		"allocated_gpu_layers":            allocatedGPULayers,
		"total_model_layers":              totalModelLayers,
		"allocated_vram_bytes":            allocatedVRAMBytes,
		"peak_context_vram_bytes":         peakContextVRAMBytes,
		"ttft_latency_seconds":            ttftLatencySeconds,
		"model_swapping_failures":         modelSwappingFailures,
		"un_gated_mutations":              unGatedMutations,
		"never_equate_intent_to_approval": g.NeverEquateIntentToApproval,
	}

	eventType := "runner_flagged_debt"
	if isProductionReady {
		eventType = "runner_authorized"
	}

	entry := g.Ledger.RecordRunnerEvent(runnerID, eventType, readiness, criticalSmells, metadata)

	return &OllamaDebtReport{
		RunnerID:                 runnerID,
		ODIScore:                 odiScore,
		ContextLayerSprawl:       math.Round(vramRatio*100) / 100,
		TTFTLatencySeconds:       math.Round(ttftLatencySeconds*100) / 100,
		MutationSafetyScore:      mutationScore,
		ProductionReadinessIndex: readiness,
		IsProductionReady:        isProductionReady,
		CriticalSmells:           criticalSmells,
		ReceiptHash:              entry["curr_hash"].(string),
	}, nil
}
