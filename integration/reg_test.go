//go:build integration

package integration

import "testing"

type integrationCase struct {
	Key   string
	Case  string
	Model string
	Run   func(t *testing.T)
}

type integrationModel struct {
	Name      string
	MinVRAMGB uint64
}

var (
	integrationCases    []integrationCase
	integrationCaseKeys = map[string]struct{}{}
	modelMinVRAMGB      = map[string]uint64{}
)

func registerIntegrationCases(cases ...integrationCase) {
	for _, c := range cases {
		if _, ok := integrationCaseKeys[c.Key]; ok {
			continue
		}
		integrationCaseKeys[c.Key] = struct{}{}
		integrationCases = append(integrationCases, c)
	}
}

func integrationTestCase(name, model string, run func(t *testing.T)) integrationCase {
	key := name
	if model != "" {
		key += "/" + model
	}
	return integrationCase{
		Key:   key,
		Case:  name,
		Model: model,
		Run:   run,
	}
}

func integrationModelTestCase(name, model string, run func(*testing.T, string)) integrationCase {
	return integrationTestCase(name, model, func(t *testing.T) {
		run(t, model)
	})
}

func integrationModelsTestCase(name string, models []string, run func(*testing.T, []string)) integrationCase {
	return integrationTestCase(name, "", func(t *testing.T) {
		run(t, models)
	})
}

func registerModelIntegrationCases(name string, models []string, run func(*testing.T, string)) {
	cases := make([]integrationCase, 0, len(models))
	for _, model := range models {
		model := model
		cases = append(cases, integrationCase{
			Key:   name + "/" + model,
			Case:  name,
			Model: model,
			Run: func(t *testing.T) {
				run(t, model)
			},
		})
	}
	registerIntegrationCases(cases...)
}

func modelNames(models []integrationModel) []string {
	names := make([]string, 0, len(models))
	for _, model := range models {
		names = append(names, model.Name)
	}
	return names
}

func registerModelMinVRAM(models []integrationModel) {
	for _, model := range models {
		if model.MinVRAMGB > 0 {
			modelMinVRAMGB[model.Name] = model.MinVRAMGB
		}
	}
}

func skipRegisteredMinVRAM(t *testing.T, model string) {
	t.Helper()
	if v, ok := modelMinVRAMGB[model]; ok {
		skipUnderMinVRAM(t, v)
	}
}

type knownIntegrationFlake struct {
	Scenario string
	Model    string
	Reason   string
}

var knownIntegrationFlakes = []knownIntegrationFlake{
	{
		Scenario: "tools-stress/multi_turn",
		Model:    "gemma4",
		Reason:   "returns an empty response on the agent-style multi-turn tool prompt",
	},
	{
		Scenario: "tools-stress/multi_turn",
		Model:    "qwen3.5:2b",
		Reason:   "returns an empty response after the tool result in the agent-style multi-turn prompt",
	},
	{
		Scenario: "vision-text",
		Model:    "qwen3.5:2b",
		Reason:   "times out instead of returning OCR text for the Ollamas image",
	},
	{
		Scenario: "vision-multiturn",
		Model:    "gemma4",
		Reason:   "counts five animals in the Ollamas image instead of four",
	},
	{
		Scenario: "vision-count",
		Model:    "gemma4",
		Reason:   "counts five animals in the docs image instead of four",
	},

	{
		Scenario: "tools-stress",
		Model:    "lfm2.5-thinking",
		Reason:   "returns text instead of tool calls with complex system prompts",
	},
	{
		Scenario: "tools-stress",
		Model:    "qwen3.5:2b",
		Reason:   "2B model too small for reliable multi-tool agent prompts",
	},
	{
		Scenario: "tools-stress",
		Model:    "qwen3-vl",
		Reason:   "vision model, extremely slow with complex tool prompts",
	},
	{
		Scenario: "tools-stress",
		Model:    "llama3.2",
		Reason:   "3B model too small for reliable multi-tool agent prompts",
	},
	{
		Scenario: "tools-stress",
		Model:    "mistral",
		Reason:   "7B v0.3 returns text instead of tool calls with complex prompts",
	},
	{
		Scenario: "tools-stress",
		Model:    "mixtral:8x22b",
		Reason:   "returns text instead of tool calls with complex prompts",
	},
	{
		Scenario: "tools-stress",
		Model:    "qwen2",
		Reason:   "returns text instead of tool calls with complex prompts",
	},
	{
		Scenario: "tools-stress",
		Model:    "granite3.3",
		Reason:   "returns text instead of tool calls with complex prompts",
	},

	{
		Scenario: "vision-multiturn",
		Model:    "gemma3",
		Reason:   "misidentifies briefcase as smartphone on turn 3",
	},
	{
		Scenario: "vision-multiturn",
		Model:    "llama3.2-vision",
		Reason:   "miscounts animals (says 3 instead of 4) on turn 2",
	},
	{
		Scenario: "vision-count",
		Model:    "llama3.2-vision",
		Reason:   "consistently miscounts (says 3 instead of 4)",
	},
	{
		Scenario: "vision-scene",
		Model:    "llama3.2-vision",
		Reason:   "3B model lacks cultural reference knowledge",
	},
	{
		Scenario: "vision-scene",
		Model:    "minicpm-v",
		Reason:   "too small for cultural reference detection",
	},
	{
		Scenario: "vision-multi-image",
		Model:    "llama3.2-vision",
		Reason:   "does not support multi-image input",
	},
}

// skipKnownIntegrationFlake skips known-bad model/scenario combinations
// unless OLLAMA_TEST_MODEL explicitly requests the model.
func skipKnownIntegrationFlake(t *testing.T, scenario, model string) {
	t.Helper()
	if testModel != "" {
		return
	}
	for _, flake := range knownIntegrationFlakes {
		if flake.Scenario == scenario && flake.Model == model {
			t.Skipf("known model/scenario flake: %s", flake.Reason)
		}
	}
}
