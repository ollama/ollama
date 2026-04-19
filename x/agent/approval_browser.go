package agent

import "fmt"

func browserToolDisplayName(toolName string) (string, bool) {
	switch toolName {
	case "web_search":
		return "Web Search", true
	case "web_fetch":
		return "Web Fetch", true
	default:
		return "", false
	}
}

func formatBrowserToolDisplay(toolName string, args map[string]any) (string, bool) {
	switch toolName {
	case "web_search":
		if query, ok := args["query"].(string); ok {
			return fmt.Sprintf("Tool: %s\nQuery: %s\nUses internet via ollama.com", ToolDisplayName(toolName), query), true
		}
	case "web_fetch":
		if url, ok := args["url"].(string); ok {
			return fmt.Sprintf("Tool: %s\nURL: %s\nUses internet via ollama.com", ToolDisplayName(toolName), url), true
		}
	}

	return "", false
}

func formatBrowserApprovalResult(label string, toolName string, args map[string]any) (string, bool) {
	displayName := ToolDisplayName(toolName)

	switch toolName {
	case "web_search":
		if query, ok := args["query"].(string); ok {
			return fmt.Sprintf("\033[1m%s:\033[0m %s: %s", label, displayName, truncateDisplayText(query, 40)), true
		}
	case "web_fetch":
		if url, ok := args["url"].(string); ok {
			return fmt.Sprintf("\033[1m%s:\033[0m %s: %s", label, displayName, truncateDisplayText(url, 50)), true
		}
	}

	return "", false
}
