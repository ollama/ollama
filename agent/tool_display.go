package agent

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

const maxToolInvocationCommandRunes = 100

// ToolDisplayName returns the user-facing label for a tool name.
func ToolDisplayName(name string) string {
	switch name {
	case "web_search":
		return "Web Search"
	case "web_fetch":
		return "Web Fetch"
	case "bash":
		return "Bash"
	case "powershell":
		return "PowerShell"
	case "read":
		return "Read"
	case "list":
		return "List"
	case "edit":
		return "Edit"
	case "skill":
		return "Skill"
	default:
		if name == "" {
			return "Tool"
		}
		return name
	}
}

// ToolInvocationLabel returns a compact user-facing label for a tool call.
func ToolInvocationLabel(name string, args map[string]any) string {
	displayName := ToolDisplayName(name)
	for _, key := range []string{"query", "url", "command", "path", "name"} {
		if value, ok := displayStringArg(args, key); ok {
			if IsShellToolName(name) && key == "command" {
				value = truncateDisplayRunes(value, maxToolInvocationCommandRunes)
			}
			return fmt.Sprintf("%s(%s)", displayName, strconv.Quote(value))
		}
	}
	if len(args) == 0 {
		return displayName
	}
	return fmt.Sprintf("%s(%s)", displayName, formatDisplayArgs(args))
}

// IsShellToolName reports whether name identifies a platform shell tool.
func IsShellToolName(name string) bool {
	return name == "bash" || name == "powershell"
}

func displayStringArg(args map[string]any, key string) (string, bool) {
	value, ok := args[key].(string)
	if !ok || strings.TrimSpace(value) == "" {
		return "", false
	}
	return value, true
}

func truncateDisplayRunes(value string, limit int) string {
	runes := []rune(value)
	if limit <= 0 || len(runes) <= limit {
		return value
	}
	return string(runes[:limit]) + "..."
}

func formatDisplayArgs(args map[string]any) string {
	keys := make([]string, 0, len(args))
	for key := range args {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		value := fmt.Sprintf("%v", args[key])
		value = truncateDisplayRunes(value, 100)
		parts = append(parts, fmt.Sprintf("%s=%s", key, strconv.Quote(value)))
	}
	return strings.Join(parts, ", ")
}
