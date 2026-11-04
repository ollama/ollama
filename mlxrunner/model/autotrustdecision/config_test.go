package autotrustdecision

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlxrunner/model"
)

func TestParseConfigChoiceDefaults(t *testing.T) {
	for _, tt := range []struct {
		name       string
		data       string
		trained    int
		maxChoices int
	}{
		{name: "compact text", data: `{"model_type":"qwen3_5_text"}`, trained: 16, maxChoices: 16},
		{name: "wide MoE", data: `{"model_type":"qwen3_5_moe"}`, trained: 256, maxChoices: 256},
		{name: "compact VL", data: `{"model_type":"qwen3_5"}`, trained: 16, maxChoices: 256},
		{name: "zero uses default", data: `{"max_choices":0}`, trained: 16, maxChoices: 16},
		{name: "zero uses VL default", data: `{"model_type":"qwen3_5","max_choices":0}`, trained: 16, maxChoices: 256},
		{name: "explicit wide inference", data: `{"max_choices":256}`, trained: 16, maxChoices: 256},
		{name: "explicit lower VL limit", data: `{"model_type":"qwen3_5","max_choices":8}`, trained: 16, maxChoices: 8},
		{name: "nested backbone config", data: `{"model_type":"qwen3_5","text_config":{"hidden_size":5120}}`, trained: 16, maxChoices: 256},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var head decisionHead
			head.Slots.Ranges = map[string][2]int{"choice": {8, 8 + tt.trained}}
			cfg, err := parseConfig([]byte(tt.data), head)
			if err != nil {
				t.Fatal(err)
			}
			if cfg.MaxChoices != tt.maxChoices {
				t.Fatalf("max_choices = %d, want %d", cfg.MaxChoices, tt.maxChoices)
			}
		})
	}
}

func TestLoadModelChoiceLimits(t *testing.T) {
	for _, tt := range []struct {
		name       string
		modelType  string
		trained    int
		override   int
		maxChoices int
	}{
		{name: "compact text default", modelType: "qwen3_5_text", trained: 16, maxChoices: 16},
		{name: "wide MoE default", modelType: "qwen3_5_moe_text", trained: 256, maxChoices: 255},
		{name: "compact VL default", modelType: "qwen3_5", trained: 16, maxChoices: 255},
		{name: "explicit wide inference", modelType: "qwen3_5_text", trained: 16, override: 256, maxChoices: 255},
		{name: "explicit lower VL limit", modelType: "qwen3_5", trained: 16, override: 8, maxChoices: 8},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_MODELS", t.TempDir())
			var head decisionHead
			head.Bias = make([]float32, 8+tt.trained)
			head.VerbalizerIDs = make([]int32, len(head.Bias))
			head.Slots.Verbalizers = make([]string, len(head.Bias))
			head.Slots.Ranges = map[string][2]int{"noul": {0, 2}, "score": {2, 8}, "choice": {8, len(head.Bias)}}
			head.Slots.Template = "bare-v1"
			if tt.trained == 256 {
				head.Slots.Template = "bare-v1-wide256"
			}
			for i := range head.VerbalizerIDs {
				head.VerbalizerIDs[i] = int32(i)
			}

			var added []map[string]any
			for i := range 256 {
				label := string(rune('A' + i))
				if i >= 26 {
					label = string([]rune{rune('A' + (i-26)/26), rune('A' + (i-26)%26)})
				}
				added = append(added, map[string]any{"id": i + 8, "content": label})
			}
			tokenizer := map[string]any{"model": map[string]any{"type": "BPE", "vocab": map[string]int{"x": 0}, "merges": []string{}}, "added_tokens": added}
			backbone := map[string]any{
				"model_type": tt.modelType, "hidden_size": 8, "num_hidden_layers": 1,
				"num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 8, "vocab_size": 264,
				"linear_num_value_heads": 1, "linear_num_key_heads": 1, "linear_key_head_dim": 8, "linear_value_head_dim": 8,
			}
			if tt.override != 0 {
				backbone["max_choices"] = tt.override
			}
			root := &model.Root{Manifest: &manifest.Manifest{}}
			for name, value := range map[string]any{
				"config.json": backbone, "decision_head.json": head, "tokenizer.json": tokenizer,
				"calibration.json": map[string]any{"per_kind": map[string]int{"choice": 1, "noul": 1, "score": 1}},
			} {
				data, err := json.Marshal(value)
				if err != nil {
					t.Fatal(err)
				}
				layer, err := manifest.NewLayer(strings.NewReader(string(data)), manifest.MediaTypeImageJSON)
				if err != nil {
					t.Fatal(err)
				}
				layer.Name = name
				root.Manifest.Layers = append(root.Manifest.Layers, layer)
			}
			loaded, err := newModel(root)
			if err != nil {
				t.Fatal(err)
			}
			m := loaded.(*Model)
			if len(m.choiceIDs) != tt.maxChoices || len(m.choiceCodes) != tt.maxChoices {
				t.Fatalf("loaded %d IDs and %d labels, want %d", len(m.choiceIDs), len(m.choiceCodes), tt.maxChoices)
			}
			q := &llm.ScoreQuestion{Type: "choice", Options: make([]string, tt.maxChoices)}
			if _, _, _, _, err := m.prompt(llm.ScoreRequest{}, q); err != nil {
				t.Fatalf("exact limit: %v", err)
			}
			q.Options = append(q.Options, "extra")
			if _, _, _, _, err := m.prompt(llm.ScoreRequest{}, q); err == nil {
				t.Fatal("accepted more choices than the resolved limit")
			}
		})
	}
}

func TestParseConfigRejectsInvalidChoices(t *testing.T) {
	for _, data := range []string{
		`{"max_choices":-1}`,
		`{"max_choices":1}`,
		`{"max_choices":257}`,
		`{"max_choices":2.5}`,
		`{"max_choices":"256"}`,
		`{"max_choices":`,
	} {
		t.Run(data, func(t *testing.T) {
			var head decisionHead
			head.Slots.Ranges = map[string][2]int{"choice": {8, 24}}
			if _, err := parseConfig([]byte(data), head); err == nil {
				t.Fatalf("accepted invalid config %s", data)
			}
		})
	}
}
