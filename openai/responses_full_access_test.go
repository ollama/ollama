package openai

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func fullAccessExecFixture(t *testing.T) map[string]any {
	t.Helper()
	// Captured from the unmodified codex-cli 0.155.0-alpha.16.3 backend.
	data, err := os.ReadFile("testdata/codex_exec_command.json")
	if err != nil {
		t.Fatal(err)
	}
	var tool map[string]any
	if err := json.Unmarshal(data, &tool); err != nil {
		t.Fatal(err)
	}
	return tool
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
		{"custom tool type", func(_, tool map[string]any) { tool["type"] = "custom" }},
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
	tool := fullAccessExecFixture(t)
	tool["description"] = tool["description"].(string) + "\n\nWindows shell guidance."
	tool["parameters"].(map[string]any)["properties"].(map[string]any)["yield_time_ms"].(map[string]any)["description"] = "Windows yield guidance."
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
}
