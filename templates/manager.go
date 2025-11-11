package templates

import (
	"fmt"
	"strings"
)

// Manager manages prompt templates
type Manager struct {
	templates map[string]*Template
}

// Template represents a prompt template
type Template struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Category    string            `json:"category"`
	Description string            `json:"description"`
	Template    string            `json:"template"`
	Variables   []string          `json:"variables"`
}

// NewManager creates a new template manager
func NewManager() *Manager {
	m := &Manager{
		templates: make(map[string]*Template),
	}
	m.loadBuiltinTemplates()
	return m
}

// Render renders a template with variables
func (m *Manager) Render(templateID string, vars map[string]string) (string, error) {
	template, ok := m.templates[templateID]
	if !ok {
		return "", fmt.Errorf("template not found: %s", templateID)
	}

	result := template.Template
	for key, value := range vars {
		placeholder := "{{" + key + "}}"
		result = strings.ReplaceAll(result, placeholder, value)
	}

	return result, nil
}

// GetTemplate returns a template by ID
func (m *Manager) GetTemplate(id string) (*Template, error) {
	template, ok := m.templates[id]
	if !ok {
		return nil, fmt.Errorf("template not found: %s", id)
	}
	return template, nil
}

// ListTemplates returns all templates
func (m *Manager) ListTemplates() []*Template {
	templates := make([]*Template, 0, len(m.templates))
	for _, t := range m.templates {
		templates = append(templates, t)
	}
	return templates
}

// loadBuiltinTemplates loads built-in templates
func (m *Manager) loadBuiltinTemplates() {
	m.templates["code-review"] = &Template{
		ID:          "code-review",
		Name:        "Code Review",
		Category:    "development",
		Description: "Review code for issues and suggestions",
		Template: `You are an expert code reviewer. Review the following {{language}} code:

{{code}}

Provide:
1. Security issues
2. Performance problems
3. Best practices violations
4. Refactoring suggestions`,
		Variables: []string{"language", "code"},
	}

	m.templates["bug-fix"] = &Template{
		ID:          "bug-fix",
		Name:        "Bug Fix",
		Category:    "development",
		Description: "Help fix a bug",
		Template: `I have a bug in my {{language}} code. Here's the error:

{{error}}

Here's the relevant code:

{{code}}

Please help me fix this bug.`,
		Variables: []string{"language", "error", "code"},
	}

	m.templates["documentation"] = &Template{
		ID:          "documentation",
		Name:        "Generate Documentation",
		Category:    "development",
		Description: "Generate documentation for code",
		Template: `Generate comprehensive documentation for the following {{language}} code:

{{code}}

Include:
- Function/method descriptions
- Parameter descriptions
- Return value descriptions
- Usage examples`,
		Variables: []string{"language", "code"},
	}
}

