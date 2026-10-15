package openai

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func fullAccessExecFixture(t *testing.T) map[string]any {
	t.Helper()
	// Captured from the unmodified codex-cli 0.155.0-alpha.16.3 backend.
	return map[string]any{
		"type":        "function",
		"name":        "exec_command",
		"description": "Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
		"strict":      false,
		"parameters": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"cmd":                 map[string]any{"type": "string", "description": "Shell command to execute."},
				"justification":       map[string]any{"type": "string", "description": "User-facing approval question for `require_escalated`; omit otherwise."},
				"login":               map[string]any{"type": "boolean", "description": "True runs the shell with -l/-i semantics; false disables them. Defaults to true."},
				"max_output_tokens":   map[string]any{"type": "number", "description": "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy."},
				"prefix_rule":         map[string]any{"type": "array", "description": "Reusable approval prefix for `cmd`, only with `sandbox_permissions: \"require_escalated\"`; for example [\"git\", \"pull\"].", "items": map[string]any{"type": "string"}},
				"sandbox_permissions": map[string]any{"type": "string", "description": "Per-command sandbox override. Defaults to `use_default`; use `require_escalated` for unsandboxed execution.", "enum": []any{"use_default", "require_escalated"}},
				"shell":               map[string]any{"type": "string", "description": "Shell binary to launch. Defaults to the user's default shell."},
				"tty":                 map[string]any{"type": "boolean", "description": "True allocates a PTY for the command; false or omitted uses plain pipes."},
				"workdir":             map[string]any{"type": "string", "description": "Working directory for the command. Defaults to the turn cwd."},
				"yield_time_ms":       map[string]any{"type": "number", "description": "Wait before yielding output. Defaults to 10000 ms; effective range is 250-30000 ms."},
			},
			"required":             []any{"cmd"},
			"additionalProperties": false,
		},
	}
}

func TestResponsesFullAccessTools(t *testing.T) {
	for _, namespace := range []string{"", "functions"} {
		for _, encoded := range []bool{false, true} {
			t.Run(namespace+map[bool]string{false: "/object", true: "/string"}[encoded], func(t *testing.T) {
				tool := fullAccessExecFixture(t)
				params := tool["parameters"].(map[string]any)
				params["required"] = []any{"cmd", "sandbox_permissions", "justification", "prefix_rule"}
				tool["future_field"] = json.Number("9007199254740993")
				var entry any = tool
				if namespace != "" {
					entry = map[string]any{"type": "namespace", "name": namespace, "description": "Built-ins", "tools": []any{tool}}
				}
				custom := fullAccessExecFixture(t)
				custom["description"] = "A caller-defined command tool."
				var metadata any = map[string]any{"sandbox_mode": "danger-full-access", "sandbox": "none"}
				if encoded {
					metadata = `{"sandbox_mode":"danger-full-access","sandbox":"none"}`
				}
				request := map[string]any{
					"model": "test", "tools": []any{entry, custom},
					"client_metadata": map[string]any{"x-codex-turn-metadata": metadata},
					"input":           []any{map[string]any{"type": "function_call", "call_id": "previous", "name": "exec_command", "arguments": `{"sandbox_permissions":"require_escalated","cmd":"old command"}`}},
					"unknown":         json.Number("9007199254740993"),
				}
				body, _ := json.Marshal(request)
				got, changed, err := NormalizeResponsesFullAccessTools(body)
				if err != nil || !changed {
					t.Fatalf("changed=%v err=%v", changed, err)
				}
				// Compare the entire request: only the built-in tool's three
				// escalation properties and required entries may change.
				properties := params["properties"].(map[string]any)
				for _, key := range []string{"sandbox_permissions", "justification", "prefix_rule"} {
					delete(properties, key)
				}
				params["required"] = []any{"cmd"}
				want, _ := json.Marshal(request)
				if !bytes.Equal(got, want) {
					t.Fatalf("got %s\nwant %s", got, want)
				}
				again, changed, err := NormalizeResponsesFullAccessTools(got)
				if err != nil || changed || !bytes.Equal(again, got) {
					t.Fatalf("second normalization changed request: changed=%v err=%v", changed, err)
				}
			})
		}
	}
}

func TestResponsesFullAccessToolsPreserveHTML(t *testing.T) {
	content := strings.Repeat("<tag>&</tag>", 10000) + strings.Repeat("a", 120000)
	const toolDescription = "<helper> & custom tool"
	request := map[string]any{
		"model": "test", "input": content,
		"client_metadata": map[string]any{"x-codex-turn-metadata": `{"sandbox_mode":"danger-full-access","sandbox":"none"}`},
		"tools": []any{fullAccessExecFixture(t), map[string]any{
			"type": "function", "name": "custom", "description": toolDescription,
			"parameters": map[string]any{"type": "object"},
		}},
	}
	var body bytes.Buffer
	encoder := json.NewEncoder(&body)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(request); err != nil {
		t.Fatal(err)
	}
	got, changed, err := NormalizeResponsesFullAccessTools(body.Bytes())
	if err != nil || !changed {
		t.Fatalf("changed=%v err=%v", changed, err)
	}
	if len(got) >= body.Len() {
		t.Errorf("removing schema fields grew the request: before=%d after=%d", body.Len(), len(got))
	}
	for _, text := range []string{content, toolDescription} {
		if !bytes.Contains(got, []byte(text)) {
			t.Error("conversation or custom tool description was HTML-escaped")
		}
	}
}

func TestResponsesFullAccessToolsPreserveOtherContracts(t *testing.T) {
	for _, tt := range []struct {
		name   string
		modify func(map[string]any, map[string]any)
	}{
		{"missing metadata", func(r, _ map[string]any) { delete(r, "client_metadata") }},
		{"malformed metadata", func(r, _ map[string]any) { r["client_metadata"] = "not an object" }},
		{"malformed turn metadata", func(r, _ map[string]any) { r["client_metadata"] = map[string]any{"x-codex-turn-metadata": "not JSON"} }},
		{"no sandbox mode", func(r, _ map[string]any) { r["client_metadata"] = map[string]any{"x-codex-turn-metadata": `{}`} }},
		{"no sandbox backend", func(r, _ map[string]any) {
			r["client_metadata"] = map[string]any{"x-codex-turn-metadata": `{"sandbox_mode":"danger-full-access"}`}
		}},
		{"custom sandbox profile", func(r, _ map[string]any) {
			r["client_metadata"] = map[string]any{"x-codex-turn-metadata": `{"sandbox_mode":"danger-full-access","sandbox":"seatbelt"}`}
		}},
		{"read only", func(r, _ map[string]any) {
			r["client_metadata"] = map[string]any{"x-codex-turn-metadata": `{"sandbox_mode":"read-only"}`}
		}},
		{"workspace write", func(r, _ map[string]any) {
			r["client_metadata"] = map[string]any{"x-codex-turn-metadata": `{"sandbox_mode":"workspace-write"}`}
		}},
		{"external sandbox", func(r, _ map[string]any) {
			r["client_metadata"] = map[string]any{"x-codex-turn-metadata": `{"sandbox_mode":"external-sandbox"}`}
		}},
		{"custom namespace", func(r, tool map[string]any) {
			r["tools"] = []any{map[string]any{"type": "namespace", "name": "application", "tools": []any{tool}}}
		}},
		{"nested custom namespace", func(r, tool map[string]any) {
			r["tools"] = []any{map[string]any{"type": "namespace", "name": "application", "tools": []any{map[string]any{"type": "namespace", "name": "functions", "tools": []any{tool}}}}}
		}},
		{"custom description", func(_, tool map[string]any) { tool["description"] = "Run my application's command." }},
		{"custom description suffix", func(_, tool map[string]any) {
			tool["description"] = tool["description"].(string) + "\n\nAlways supply justification for the application's audit log."
		}},
		{"custom tool type", func(_, tool map[string]any) { tool["type"] = "custom" }},
		{"strict custom contract", func(_, tool map[string]any) {
			tool["strict"] = true
			parameters := tool["parameters"].(map[string]any)
			required := []any{}
			for name := range parameters["properties"].(map[string]any) {
				required = append(required, name)
			}
			parameters["required"] = required
		}},
		{"custom functions namespace", func(r, tool map[string]any) {
			tool["parameters"].(map[string]any)["properties"].(map[string]any)["justification"].(map[string]any)["description"] = "Audit reason required by the application."
			r["tools"] = []any{map[string]any{"type": "namespace", "name": "functions", "tools": []any{tool}}}
		}},
		{"custom permission semantics", func(_, tool map[string]any) {
			tool["parameters"].(map[string]any)["properties"].(map[string]any)["sandbox_permissions"].(map[string]any)["default"] = "require_escalated"
		}},
		{"custom prefix semantics", func(_, tool map[string]any) {
			tool["parameters"].(map[string]any)["properties"].(map[string]any)["prefix_rule"].(map[string]any)["description"] = "Prefix required to select the application's command handler."
		}},
		{"required by composed schema", func(_, tool map[string]any) {
			tool["parameters"].(map[string]any)["allOf"] = []any{map[string]any{"required": []any{"justification"}}}
		}},
		{"reference to removed property", func(_, tool map[string]any) {
			tool["parameters"].(map[string]any)["properties"].(map[string]any)["workdir"] = map[string]any{"$ref": "#/properties/justification"}
		}},
		{"custom command parameter", func(_, tool map[string]any) {
			tool["parameters"].(map[string]any)["properties"].(map[string]any)["host"] = map[string]any{"type": "string", "description": "Remote command host."}
		}},
		{"incomplete command contract", func(_, tool map[string]any) {
			delete(tool["parameters"].(map[string]any)["properties"].(map[string]any), "shell")
		}},
		{"custom required parameter", func(_, tool map[string]any) {
			tool["parameters"].(map[string]any)["required"] = []any{"cmd", "workdir"}
		}},
		{"custom workdir semantics", func(_, tool map[string]any) {
			tool["parameters"].(map[string]any)["properties"].(map[string]any)["workdir"].(map[string]any)["description"] = "Select the tenant whose command service should run."
		}},
		{"another function", func(_, tool map[string]any) { tool["name"] = "shell_command" }},
		{"custom command schema", func(_, tool map[string]any) {
			tool["parameters"].(map[string]any)["properties"].(map[string]any)["cmd"] = map[string]any{"type": "object"}
		}},
		{"other permission contract", func(_, tool map[string]any) {
			tool["parameters"].(map[string]any)["properties"].(map[string]any)["sandbox_permissions"] = map[string]any{"type": "string", "enum": []any{"use_default", "require_escalated", "with_additional_permissions"}}
		}},
		{"multiple environments", func(_, tool map[string]any) {
			tool["parameters"].(map[string]any)["properties"].(map[string]any)["environment_id"] = map[string]any{"type": "string"}
		}},
		{"missing parameters", func(_, tool map[string]any) { delete(tool, "parameters") }},
		{"malformed parameters", func(_, tool map[string]any) { tool["parameters"] = 42 }},
		{"no tools", func(r, _ map[string]any) { delete(r, "tools") }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tool := fullAccessExecFixture(t)
			request := map[string]any{"model": "test", "tools": []any{tool}, "client_metadata": map[string]any{"x-codex-turn-metadata": `{"sandbox_mode":"danger-full-access","sandbox":"none"}`}}
			tt.modify(request, tool)
			body, _ := json.Marshal(request)
			got, changed, err := NormalizeResponsesFullAccessTools(body)
			if err != nil || changed || !bytes.Equal(got, body) {
				t.Fatalf("request changed: changed=%v err=%v\ngot %s\nwant %s", changed, err, got, body)
			}
		})
	}
}

func TestResponsesFullAccessToolsWindowsDescription(t *testing.T) {
	// Apply the two differences from Codex's native Windows schema to the base
	// fixture. Guidance follows the executor OS; yield limits follow the host OS.
	for _, host := range []string{"windows", "other"} {
		t.Run(host, func(t *testing.T) {
			tool := fullAccessExecFixture(t)
			tool["description"] = tool["description"].(string) + "\n\nWindows safety rules:\n" +
				"- Do not compose destructive filesystem commands across shells. Do not enumerate paths in PowerShell and then pass them to `cmd /c`, batch builtins, or another shell for deletion or moving. Use one shell end-to-end, prefer native PowerShell cmdlets such as `Remove-Item` / `Move-Item` with `-LiteralPath`, and avoid string-built shell commands for file operations.\n" +
				"- Before any recursive delete or move on Windows, verify the resolved absolute target paths stay within the intended workspace or explicitly named target directory. Never issue a recursive delete or move against a computed path if the final target has not been checked.\n" +
				"- When using `Start-Process` to launch a background helper or service, pass `-WindowStyle Hidden` unless the user explicitly asked for a visible interactive window. Use visible windows only for interactive tools the user needs to see or control."
			if host == "windows" {
				tool["parameters"].(map[string]any)["properties"].(map[string]any)["yield_time_ms"].(map[string]any)["description"] = "Maximum time to wait before returning a session ID for a still-running command. Commands that finish sooner return immediately. For ordinary commands, omit this parameter to use the 10000 ms default. Effective range on Windows is 10000-30000 ms."
			}
			body, _ := json.Marshal(map[string]any{"tools": []any{tool}, "client_metadata": map[string]any{"x-codex-turn-metadata": `{"sandbox_mode":"danger-full-access","sandbox":"none"}`}})
			got, changed, err := NormalizeResponsesFullAccessTools(body)
			if err != nil || !changed {
				t.Fatalf("changed=%v err=%v", changed, err)
			}
			var request struct {
				Tools []ResponsesTool `json:"tools"`
			}
			if err := json.Unmarshal(got, &request); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(request.Tools[0].Parameters["required"], []any{"cmd"}) {
				t.Fatal("required command argument changed")
			}
			for _, key := range []string{"sandbox_permissions", "justification", "prefix_rule"} {
				if _, exists := request.Tools[0].Parameters["properties"].(map[string]any)[key]; exists {
					t.Errorf("%s was not filtered", key)
				}
			}
			tool["description"] = tool["description"].(string) + "\n\nAlways supply justification for the application's audit log."
			body, _ = json.Marshal(map[string]any{"tools": []any{tool}, "client_metadata": map[string]any{"x-codex-turn-metadata": `{"sandbox_mode":"danger-full-access","sandbox":"none"}`}})
			got, changed, err = NormalizeResponsesFullAccessTools(body)
			if err != nil || changed || !bytes.Equal(got, body) {
				t.Fatalf("custom Windows contract changed: changed=%v err=%v", changed, err)
			}
		})
	}
}

func TestResponsesFullAccessToolsWithoutTTY(t *testing.T) {
	tool := fullAccessExecFixture(t)
	// Codex omits tty when PTY allocation is disabled; the remaining built-in
	// command contract and permission handling are unchanged.
	delete(tool["parameters"].(map[string]any)["properties"].(map[string]any), "tty")
	body, _ := json.Marshal(map[string]any{"tools": []any{tool}, "client_metadata": map[string]any{"x-codex-turn-metadata": `{"sandbox_mode":"danger-full-access","sandbox":"none"}`}})
	_, changed, err := NormalizeResponsesFullAccessTools(body)
	if err != nil || !changed {
		t.Fatalf("changed=%v err=%v", changed, err)
	}
}
