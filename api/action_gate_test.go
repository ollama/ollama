package api

import (
	"testing"
)

func TestActionGateAPIGuardrail_VerifyToolAction(t *testing.T) {
	guardrail := NewActionGateAPIGuardrail()

	// 1. Non-destructive tool execution
	allowed, hash1, err := guardrail.VerifyToolAction("llama3:latest", "web_search", false, "")
	if err != nil || !allowed || hash1 == "" {
		t.Fatalf("Expected non-destructive tool to pass, got err: %v", err)
	}

	// 2. Destructive tool execution without prove token fails
	allowed, _, err = guardrail.VerifyToolAction("llama3:latest", "delete_file", true, "")
	if err == nil || allowed {
		t.Fatalf("Expected destructive tool without prove token to fail, got allowed")
	}

	// 3. Destructive tool execution with valid prove token passes
	allowed, hash3, err := guardrail.VerifyToolAction("llama3:latest", "delete_file", true, "prov_live_1234567890abcdef1234567890abcdef")
	if err != nil || !allowed || hash3 == "" {
		t.Fatalf("Expected authorized prove token to pass, got err: %v", err)
	}

	// 4. Verify cryptographic hash chain integrity
	entries := guardrail.Ledger.GetEntries()
	if len(entries) != 3 {
		t.Fatalf("Expected 3 ledger entries, got %d", len(entries))
	}
	if !guardrail.Ledger.VerifyIntegrity() {
		t.Fatalf("Expected ledger integrity verification to pass")
	}
}
