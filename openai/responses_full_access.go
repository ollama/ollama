package openai

import (
	"bytes"
	"encoding/json"
	"slices"
)

// NormalizeResponsesFullAccessTools preserves the Codex gateway's Full Access
// tool contract on the standard Responses endpoint. This only changes the schema
// advertised to the model; the client still owns command execution and approval.
func NormalizeResponsesFullAccessTools(body []byte) ([]byte, bool, error) {
	var request map[string]json.RawMessage
	if err := json.Unmarshal(body, &request); err != nil {
		return nil, false, err
	}
	tools := codexFullAccessTools(request)
	changed := false
	for _, value := range tools {
		tool, _ := value.(map[string]any)
		if tool["type"] == "namespace" {
			// Codex may group its built-ins under functions. The namespace alone
			// does not identify a built-in; each member must match its contract.
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
	request["tools"], err = marshalNoHTMLEscape(tools)
	if err != nil {
		return nil, false, err
	}
	normalized, err := marshalNoHTMLEscape(request)
	return normalized, err == nil, err
}

// marshalNoHTMLEscape avoids expanding HTML-like conversation or tool content
// into escape sequences that count against upstream request byte limits.
func marshalNoHTMLEscape(value any) ([]byte, error) {
	var body bytes.Buffer
	encoder := json.NewEncoder(&body)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(body.Bytes(), []byte("\n")), nil
}

// Unsupported metadata and tool declarations are left to the normal Responses
// validation path. They do not opt a caller into this compatibility behavior.
func codexFullAccessTools(request map[string]json.RawMessage) []any {
	var metadata map[string]json.RawMessage
	if json.Unmarshal(request["client_metadata"], &metadata) != nil {
		return nil
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
		return nil
	}
	// Keep numbers intact, including custom tools sharing the request.
	var tools []any
	decoder := json.NewDecoder(bytes.NewReader(request["tools"]))
	decoder.UseNumber()
	if decoder.Decode(&tools) != nil {
		return nil
	}
	return tools
}

func normalizeCodexFullAccessExec(tool map[string]any) bool {
	const description = "Runs a command in a PTY, returning output or a session ID for ongoing interaction."
	const windowsDescription = description + "\n\nWindows safety rules:\n" +
		"- Do not compose destructive filesystem commands across shells. Do not enumerate paths in PowerShell and then pass them to `cmd /c`, batch builtins, or another shell for deletion or moving. Use one shell end-to-end, prefer native PowerShell cmdlets such as `Remove-Item` / `Move-Item` with `-LiteralPath`, and avoid string-built shell commands for file operations.\n" +
		"- Before any recursive delete or move on Windows, verify the resolved absolute target paths stay within the intended workspace or explicitly named target directory. Never issue a recursive delete or move against a computed path if the final target has not been checked.\n" +
		"- When using `Start-Process` to launch a background helper or service, pass `-WindowStyle Hidden` unless the user explicitly asked for a visible interactive window. Use visible windows only for interactive tools the user needs to see or control."
	text, _ := tool["description"].(string)
	if tool["type"] != "function" || tool["name"] != "exec_command" || tool["strict"] != false ||
		(text != description && text != windowsDescription) {
		return false
	}
	parameters, _ := tool["parameters"].(map[string]any)
	properties, _ := parameters["properties"].(map[string]any)
	required, _ := parameters["required"].([]any)
	// Composed schemas or references may still require a property after it is
	// removed. Only the simple, known command contract can be rewritten safely.
	if len(parameters) != 4 || parameters["type"] != "object" || parameters["additionalProperties"] != false || !slices.Contains(required, any("cmd")) {
		return false
	}
	for _, name := range required {
		switch name {
		case "cmd", "sandbox_permissions", "justification", "prefix_rule":
		default:
			return false
		}
	}
	// Codex can disable PTY allocation and omit tty. All other parameters must
	// match the known contract. This also excludes environment_id: turn metadata
	// cannot establish the permission profile of another execution environment.
	expectedCount := 10
	if _, ok := properties["tty"]; !ok {
		expectedCount--
	}
	if len(properties) != expectedCount {
		return false
	}
	for _, expected := range []struct {
		name, kind, description string
	}{
		{"cmd", "string", "Shell command to execute."},
		{"workdir", "string", "Working directory for the command. Defaults to the turn cwd."},
		{"shell", "string", "Shell binary to launch. Defaults to the user's default shell."},
		{"login", "boolean", "True runs the shell with -l/-i semantics; false disables them. Defaults to true."},
		{"tty", "boolean", "True allocates a PTY for the command; false or omitted uses plain pipes."},
		{"yield_time_ms", "number", "Wait before yielding output. Defaults to 10000 ms; effective range is 250-30000 ms."},
		{"max_output_tokens", "number", "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy."},
	} {
		value, exists := properties[expected.name]
		if expected.name == "tty" && !exists {
			continue
		}
		property, _ := value.(map[string]any)
		matchesDescription := property["description"] == expected.description
		if expected.name == "yield_time_ms" && property["description"] == "Maximum time to wait before returning a session ID for a still-running command. Commands that finish sooner return immediately. For ordinary commands, omit this parameter to use the 10000 ms default. Effective range on Windows is 10000-30000 ms." {
			matchesDescription = true
		}
		if len(property) != 2 || property["type"] != expected.kind || !matchesDescription {
			return false
		}
	}
	permissions, _ := properties["sandbox_permissions"].(map[string]any)
	justification, _ := properties["justification"].(map[string]any)
	prefix, _ := properties["prefix_rule"].(map[string]any)
	items, _ := prefix["items"].(map[string]any)
	values, _ := permissions["enum"].([]any)
	if len(permissions) != 3 || permissions["type"] != "string" || len(values) != 2 ||
		permissions["description"] != "Per-command sandbox override. Defaults to `use_default`; use `require_escalated` for unsandboxed execution." ||
		!slices.Contains(values, any("use_default")) || !slices.Contains(values, any("require_escalated")) ||
		len(justification) != 2 || justification["type"] != "string" ||
		justification["description"] != "User-facing approval question for `require_escalated`; omit otherwise." ||
		len(prefix) != 3 || prefix["type"] != "array" || len(items) != 1 || items["type"] != "string" ||
		prefix["description"] != "Reusable approval prefix for `cmd`, only with `sandbox_permissions: \"require_escalated\"`; for example [\"git\", \"pull\"]." {
		return false
	}
	parameters["required"], _ = RemoveExecEscalationParameters(properties, required)
	return true
}

// RemoveExecEscalationParameters removes escalation properties and their required
// entries. Callers must establish Full Access and tool eligibility first. The
// value types preserve each caller's decoded schema and required representation.
func RemoveExecEscalationParameters[P, R any](properties map[string]P, required []R) ([]R, bool) {
	changed := false
	for _, name := range []string{"sandbox_permissions", "justification", "prefix_rule"} {
		if _, exists := properties[name]; exists {
			delete(properties, name)
			changed = true
		}
	}
	if !changed {
		return required, false
	}
	return slices.DeleteFunc(required, func(value R) bool {
		switch any(value) {
		case "sandbox_permissions", "justification", "prefix_rule":
			return true
		default:
			return false
		}
	}), true
}
