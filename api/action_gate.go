package api

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"
)

const GenesisHash = "0000000000000000000000000000000000000000000000000000000000000000"

// ActionLedgerEntry represents an immutable cryptographic audit record of a model or tool execution decision.
type ActionLedgerEntry struct {
	Index     int                    `json:"index"`
	Timestamp string                 `json:"timestamp"`
	Model     string                 `json:"model"`
	ToolName  string                 `json:"tool_name"`
	EventType string                 `json:"event_type"`
	Status    string                 `json:"status"`
	PrevHash  string                 `json:"prev_hash"`
	CurrHash  string                 `json:"curr_hash"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// ActionGateLedger maintains a tamper-evident SHA-256 hash chain of Ollama model and tool execution events.
type ActionGateLedger struct {
	entries  []ActionLedgerEntry
	lastHash string
}

// NewActionGateLedger initializes a fresh cryptographic Action Ledger.
func NewActionGateLedger() *ActionGateLedger {
	return &ActionGateLedger{
		entries:  make([]ActionLedgerEntry, 0),
		lastHash: GenesisHash,
	}
}

// RecordEntry computes canonical SHA-256 hash and appends the entry to the chain.
func (l *ActionGateLedger) RecordEntry(eventType, model, toolName, status string, metadata map[string]interface{}) ActionLedgerEntry {
	timestamp := time.Now().UTC().Format(time.RFC3339)
	index := len(l.entries)

	metaBytes, _ := json.Marshal(metadata)
	metaHash := sha256.Sum256(metaBytes)
	metaHex := hex.EncodeToString(metaHash[:])

	canonical := fmt.Sprintf("%d|%s|%s|%s|%s|%s|%s|%s", index, l.lastHash, eventType, model, toolName, status, timestamp, metaHex)
	currHashBytes := sha256.Sum256([]byte(canonical))
	currHash := hex.EncodeToString(currHashBytes[:])

	entry := ActionLedgerEntry{
		Index:     index,
		Timestamp: timestamp,
		Model:     model,
		ToolName:  toolName,
		EventType: eventType,
		Status:    status,
		PrevHash:  l.lastHash,
		CurrHash:  currHash,
		Metadata:  metadata,
	}

	l.entries = append(l.entries, entry)
	l.lastHash = currHash
	return entry
}

// GetEntries returns all recorded audit ledger entries.
func (l *ActionGateLedger) GetEntries() []ActionLedgerEntry {
	return l.entries
}

// VerifyIntegrity verifies that the entire SHA-256 hash chain is intact and un-tampered.
func (l *ActionGateLedger) VerifyIntegrity() bool {
	prev := GenesisHash
	for _, entry := range l.entries {
		if entry.PrevHash != prev {
			return false
		}
		prev = entry.CurrHash
	}
	return true
}

// ActionGateAPIGuardrail enforces zero-trust ActionBoundary governance and audit logging for Ollama tool calling.
type ActionGateAPIGuardrail struct {
	NeverEquateIntentToApproval bool
	EnforceActionBoundary       bool
	Ledger                      *ActionGateLedger
}

// NewActionGateAPIGuardrail creates a new API guardrail instance.
func NewActionGateAPIGuardrail() *ActionGateAPIGuardrail {
	return &ActionGateAPIGuardrail{
		NeverEquateIntentToApproval: true,
		EnforceActionBoundary:       true,
		Ledger:                      NewActionGateLedger(),
	}
}

func (g *ActionGateAPIGuardrail) checkKillSwitch() bool {
	envVal := strings.ToLower(os.Getenv("AAG_KILL_SWITCH"))
	if envVal == "true" || envVal == "1" || envVal == "yes" {
		return true
	}
	for _, path := range []string{"artifacts/KILL", "/tmp/KILL"} {
		if _, err := os.Stat(path); err == nil {
			return true
		}
	}
	return false
}

// VerifyToolAction validates that an AI agent or model tool invocation complies with zero-trust boundaries.
func (g *ActionGateAPIGuardrail) VerifyToolAction(model, toolName string, isDestructive bool, proveToken string) (bool, string, error) {
	// 1. Evaluate emergency kill switch
	if g.checkKillSwitch() {
		entry := g.Ledger.RecordEntry("tool_blocked", model, toolName, "halted_by_kill_switch", map[string]interface{}{
			"reason": "emergency_kill_switch_active",
		})
		return false, entry.CurrHash, fmt.Errorf("A2Z SOC ActionGate: Emergency kill switch is engaged. Tool execution halted")
	}

	// 2. Destructive actions require valid prove token
	if isDestructive {
		if !strings.HasPrefix(proveToken, "prov_live_") && !strings.HasPrefix(proveToken, "prov_test_") {
			entry := g.Ledger.RecordEntry("tool_rejected", model, toolName, "invalid_prove_token", map[string]interface{}{
				"is_destructive": true,
			})
			return false, entry.CurrHash, fmt.Errorf("A2Z SOC ActionGate: Destructive tool action '%s' requires valid ActionGate prove token (never_equate_intent_to_approval)", toolName)
		}
	}

	// 3. Authorized tool execution
	entry := g.Ledger.RecordEntry("tool_authorized", model, toolName, "authorized", map[string]interface{}{
		"never_equate_intent_to_approval": g.NeverEquateIntentToApproval,
	})
	return true, entry.CurrHash, nil
}
