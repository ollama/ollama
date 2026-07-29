package tools

import (
	"context"
	"errors"

	"github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

// Skill is the model-facing adapter for the core agent skill catalog.
// Model-initiated loads require approval because a skill's instructions can
// influence the rest of the run. Explicit user activation is handled by the
// session's synthetic skill call and bypasses this adapter.
type Skill struct{ Catalog *agent.SkillCatalog }

func (t *Skill) Name() string { return "skill" }

func (t *Skill) Description() string {
	return "Load a named Ollama skill and return its instructions."
}

func (t *Skill) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("name", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "Name of the skill to load."})
	return api.ToolFunction{Name: t.Name(), Description: t.Description(), Parameters: api.ToolFunctionParameters{Type: "object", Properties: props, Required: []string{"name"}}}
}

func (t *Skill) RequiresApproval(map[string]any) bool { return true }

func (t *Skill) Execute(_ context.Context, _ agent.ToolContext, args map[string]any) (agent.ToolResult, error) {
	name, ok := args["name"].(string)
	if !ok {
		return agent.ToolResult{}, errors.New("name parameter is required")
	}
	skill, err := t.Catalog.Load(name)
	if err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: skill.Content()}, nil
}
