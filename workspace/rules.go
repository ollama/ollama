package workspace

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
)

// RulesManager manages rules.md file
type RulesManager struct {
	manager *Manager
}

// NewRulesManager creates rules manager
func NewRulesManager(m *Manager) *RulesManager {
	return &RulesManager{manager: m}
}

// GetRules reads and parses rules.md
func (rm *RulesManager) GetRules(workspacePath string) (*Rules, error) {
	rulesPath := filepath.Join(workspacePath, ".leah", "rules.md")

	content, err := os.ReadFile(rulesPath)
	if err != nil {
		return nil, err
	}

	return rm.parseRules(string(content)), nil
}

// Rules represents parsed rules
type Rules struct {
	Prohibitions  []string `json:"prohibitions"`
	Requirements  []string `json:"requirements"`
	CodeStyle     []string `json:"code_style"`
	RawContent    string   `json:"raw_content"`
}

// parseRules parses rules.md content
func (rm *RulesManager) parseRules(content string) *Rules {
	rules := &Rules{
		RawContent: content,
	}

	scanner := bufio.NewScanner(strings.NewReader(content))
	var currentSection string

	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		// Detect sections
		if strings.Contains(trimmed, "YASAKLAR") || strings.Contains(trimmed, "PROHIBIT") {
			currentSection = "prohibitions"
			continue
		} else if strings.Contains(trimmed, "ZORUNLU") || strings.Contains(trimmed, "REQUIRE") {
			currentSection = "requirements"
			continue
		} else if strings.Contains(trimmed, "KOD") || strings.Contains(trimmed, "CODE") {
			currentSection = "code_style"
			continue
		}

		// Parse bullet points
		if strings.HasPrefix(trimmed, "- ") {
			rule := strings.TrimPrefix(trimmed, "- ")
			switch currentSection {
			case "prohibitions":
				rules.Prohibitions = append(rules.Prohibitions, rule)
			case "requirements":
				rules.Requirements = append(rules.Requirements, rule)
			case "code_style":
				rules.CodeStyle = append(rules.CodeStyle, rule)
			}
		}
	}

	return rules
}

// ToSystemPrompt converts rules to system prompt
func (r *Rules) ToSystemPrompt() string {
	var parts []string

	parts = append(parts, "# WORKSPACE RULES")
	parts = append(parts, "You must strictly follow these rules defined by the user:")
	parts = append(parts, "")

	if len(r.Prohibitions) > 0 {
		parts = append(parts, "## PROHIBITIONS (NEVER DO THESE):")
		for _, rule := range r.Prohibitions {
			parts = append(parts, "- "+rule)
		}
		parts = append(parts, "")
	}

	if len(r.Requirements) > 0 {
		parts = append(parts, "## REQUIREMENTS (ALWAYS DO THESE):")
		for _, rule := range r.Requirements {
			parts = append(parts, "- "+rule)
		}
		parts = append(parts, "")
	}

	return strings.Join(parts, "\n")
}

// UpdateRules updates rules.md file
func (rm *RulesManager) UpdateRules(workspacePath string, content string) error {
	rulesPath := filepath.Join(workspacePath, ".leah", "rules.md")
	return os.WriteFile(rulesPath, []byte(content), 0644)
}
