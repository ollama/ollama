package create

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// hindsightFormatID names Hindsight's decision checkpoints: a causal Qwen3.5
// decoder that reads every question of a request in one sequence and answers
// each at its "Qn:" slot through the tied LM head's option-code rows.
const hindsightFormatID = "hindsight-decision-v1"

// prepareHindsightInventory checks the decision config. The checkpoint has no
// readout tensor: answers come from the backbone's own LM head.
func prepareHindsightInventory(inv Inventory) (Inventory, error) {
	data, err := os.ReadFile(filepath.Join(inv.Dir, "decision_config.json"))
	if err != nil {
		return Inventory{}, fmt.Errorf("read Hindsight decision config: %w", err)
	}
	var decision struct {
		FormatVersion  int    `json:"format_version"`
		FormatID       string `json:"format_id"`
		PromptVersion  int    `json:"prompt_version"`
		Codes          string `json:"codes"`
		QuestionLabels string `json:"question_labels"`
		PlainText      bool   `json:"plain_text"`
		AttentionMode  string `json:"attention_mode"`
	}
	if err := json.Unmarshal(data, &decision); err != nil {
		return Inventory{}, fmt.Errorf("parse Hindsight decision config: %w", err)
	}
	if decision.FormatVersion != 1 || decision.FormatID != hindsightFormatID || decision.PromptVersion != 2 ||
		decision.Codes != "pairs" || decision.QuestionLabels != "positions" || !decision.PlainText || decision.AttentionMode != "causal" {
		return Inventory{}, fmt.Errorf("unsupported Hindsight decision config (format %q, prompt version %d)", decision.FormatID, decision.PromptVersion)
	}
	return inv, nil
}
