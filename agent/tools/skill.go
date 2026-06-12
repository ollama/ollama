package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/agent/skills"
	"github.com/ollama/ollama/api"
)

type Skill struct {
	catalog *skills.Catalog
}

func NewSkill(catalog *skills.Catalog) *Skill {
	return &Skill{catalog: catalog}
}

func (s *Skill) Name() string {
	return "skill"
}

func (s *Skill) Description() string {
	return "Load the full SKILL.md instructions for an installed agent skill by name."
}

func (s *Skill) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("name", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "Name of the skill to load.",
	})
	return api.ToolFunction{
		Name:        s.Name(),
		Description: s.Description(),
		Parameters: api.ToolFunctionParameters{
			Type:       "object",
			Required:   []string{"name"},
			Properties: props,
		},
	}
}

func (s *Skill) Execute(_ context.Context, _ agent.ToolContext, args map[string]any) (agent.ToolResult, error) {
	name, _ := args["name"].(string)
	name = skills.NormalizeName(name)
	if name == "" {
		return agent.ToolResult{}, fmt.Errorf("name parameter is required")
	}
	if s.catalog == nil || s.catalog.Empty() {
		return agent.ToolResult{}, fmt.Errorf("no skills are installed")
	}

	skill, ok := s.catalog.Find(name)
	if !ok {
		return agent.ToolResult{}, fmt.Errorf("unknown skill: %s", name)
	}
	content, err := skill.Read()
	if err != nil {
		return agent.ToolResult{}, err
	}

	var b strings.Builder
	b.WriteString("Loaded skill: ")
	b.WriteString(skill.Name)
	b.WriteByte('\n')
	b.WriteString("Skill directory: ")
	b.WriteString(skill.Dir)
	b.WriteString("\nResolve relative file references from the skill directory.\n\n")
	b.WriteString(content)
	return agent.ToolResult{Content: b.String()}, nil
}
