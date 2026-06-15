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
	content, err := SkillResultContent(skill)
	if err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: content}, nil
}

func ManualSkillMessages(skill skills.Skill, request string, ordinal int) ([]api.Message, error) {
	content, err := SkillResultContent(skill)
	if err != nil {
		return nil, err
	}

	args := api.NewToolCallFunctionArguments()
	args.Set("name", skill.Name)
	callID := manualSkillToolCallID(skill.Name, ordinal)

	userContent := strings.TrimSpace(request)
	if userContent == "" {
		userContent = fmt.Sprintf("Use the %s skill.", skill.Name)
	}

	return []api.Message{
		{Role: "user", Content: userContent},
		{
			Role: "assistant",
			ToolCalls: []api.ToolCall{{
				ID: callID,
				Function: api.ToolCallFunction{
					Name:      "skill",
					Arguments: args,
				},
			}},
		},
		{Role: "tool", ToolName: "skill", ToolCallID: callID, Content: content},
	}, nil
}

func SkillResultContent(skill skills.Skill) (string, error) {
	content, err := skill.Read()
	if err != nil {
		return "", err
	}

	var b strings.Builder
	b.WriteString("Loaded skill: ")
	b.WriteString(skill.Name)
	b.WriteByte('\n')
	b.WriteString("Skill directory: ")
	b.WriteString(skill.Dir)
	b.WriteString("\nResolve relative file references from the skill directory.\n\n")
	b.WriteString(content)
	return b.String(), nil
}

func manualSkillToolCallID(skillName string, ordinal int) string {
	name := strings.Trim(strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z':
			return r
		case r >= 'A' && r <= 'Z':
			return r
		case r >= '0' && r <= '9':
			return r
		case r == '-' || r == '_':
			return r
		default:
			return '-'
		}
	}, skillName), "-")
	if name == "" {
		name = "skill"
	}
	if ordinal <= 0 {
		return "manual-skill-" + name
	}
	return fmt.Sprintf("manual-skill-%d-%s", ordinal, name)
}
