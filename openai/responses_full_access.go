package openai

import (
	"bytes"
	"encoding/json"
	"slices"
	"strings"
)

// NormalizeResponsesFullAccessTools preserves the Codex gateway's Full Access
// tool contract on the standard Responses endpoint. This only changes the schema
// advertised to the model; the client still owns command execution and approval.
func NormalizeResponsesFullAccessTools(body []byte) ([]byte, bool, error) {
	var request map[string]json.RawMessage
	if err := json.Unmarshal(body, &request); err != nil {
		return nil, false, err
	}
	var metadata map[string]json.RawMessage
	if json.Unmarshal(request["client_metadata"], &metadata) != nil {
		return body, false, nil
	}
	turn := metadata["x-codex-turn-metadata"]
	var encoded string
	if json.Unmarshal(turn, &encoded) == nil {
		turn = []byte(encoded)
	}
	var policy struct {
		SandboxMode string `json:"sandbox_mode"`
		Sandbox     string `json:"sandbox"`
	}
	// The mode alone can also describe a custom profile with full disk access
	// and a restricted network. Require the reported sandbox to be disabled too.
	if json.Unmarshal(turn, &policy) != nil || policy.SandboxMode != "danger-full-access" || policy.Sandbox != "none" {
		return body, false, nil
	}

	// Keep unknown schema fields and numbers intact, including custom tools that
	// share a request with the built-in command tool.
	var tools []any
	decoder := json.NewDecoder(bytes.NewReader(request["tools"]))
	decoder.UseNumber()
	if decoder.Decode(&tools) != nil {
		return body, false, nil
	}
	changed := false
	for _, value := range tools {
		tool, _ := value.(map[string]any)
		if tool["type"] == "namespace" {
			// Other namespaces belong to caller-defined tools, even if their
			// members have the same names and schemas as Codex's built-ins.
			if tool["name"] == "functions" {
				members, _ := tool["tools"].([]any)
				for _, member := range members {
					member, _ := member.(map[string]any)
					changed = normalizeCodexFullAccessExec(member) || changed
				}
			}
			continue
		}
		changed = normalizeCodexFullAccessExec(tool) || changed
	}
	if !changed {
		return body, false, nil
	}
	var err error
	request["tools"], err = json.Marshal(tools)
	if err != nil {
		return nil, false, err
	}
	normalized, err := json.Marshal(request)
	return normalized, err == nil, err
}

func normalizeCodexFullAccessExec(tool map[string]any) bool {
	const description = "Runs a command in a PTY, returning output or a session ID for ongoing interaction."
	text, _ := tool["description"].(string)
	if tool["type"] != "function" || tool["name"] != "exec_command" ||
		(text != description && !strings.HasPrefix(text, description+"\n\n")) {
		return false
	}
	parameters, _ := tool["parameters"].(map[string]any)
	properties, _ := parameters["properties"].(map[string]any)
	required, _ := parameters["required"].([]any)
	if parameters["type"] != "object" || parameters["additionalProperties"] != false || !slices.Contains(required, any("cmd")) {
		return false
	}
	// Turn metadata describes the primary environment. Do not infer the policy
	// of another environment or a newer, granular permission contract.
	if _, ok := properties["environment_id"]; ok {
		return false
	}
	cmd, _ := properties["cmd"].(map[string]any)
	permissions, _ := properties["sandbox_permissions"].(map[string]any)
	justification, _ := properties["justification"].(map[string]any)
	prefix, _ := properties["prefix_rule"].(map[string]any)
	items, _ := prefix["items"].(map[string]any)
	values, _ := permissions["enum"].([]any)
	if cmd["type"] != "string" || cmd["description"] != "Shell command to execute." ||
		permissions["type"] != "string" || len(values) != 2 ||
		!slices.Contains(values, any("use_default")) || !slices.Contains(values, any("require_escalated")) ||
		justification["type"] != "string" || prefix["type"] != "array" || items["type"] != "string" {
		return false
	}
	for _, name := range []string{"sandbox_permissions", "justification", "prefix_rule"} {
		delete(properties, name)
	}
	parameters["required"] = slices.DeleteFunc(required, func(value any) bool {
		return value == "sandbox_permissions" || value == "justification" || value == "prefix_rule"
	})
	return true
}
