package agent

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// ToolDisplayName returns the user-facing label for a tool name.
func ToolDisplayName(name string) string {
	switch name {
	case "web_search":
		return "Web Search"
	case "web_fetch":
		return "Web Fetch"
	case "bash":
		return "Bash"
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
			return fmt.Sprintf("%s(%s)", displayName, strconv.Quote(value))
		}
	}
	if len(args) == 0 {
		return displayName
	}
	return fmt.Sprintf("%s(%s)", displayName, formatDisplayArgs(args))
}

func displayStringArg(args map[string]any, key string) (string, bool) {
	value, ok := args[key].(string)
	if !ok || strings.TrimSpace(value) == "" {
		return "", false
	}
	return value, true
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
		if len([]rune(value)) > 100 {
			value = string([]rune(value)[:100]) + "..."
		}
		parts = append(parts, fmt.Sprintf("%s=%s", key, strconv.Quote(value)))
	}
	return strings.Join(parts, ", ")
}
