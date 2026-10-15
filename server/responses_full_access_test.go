package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/klauspost/compress/zstd"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/middleware"
)

func fullAccessRouteBody(t *testing.T, model, mode, namespace string, stream bool) []byte {
	t.Helper()
	// Captured from the unmodified codex-cli 0.155.0-alpha.16.3 backend.
	var entry any = map[string]any{
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
	if namespace != "" {
		entry = map[string]any{"type": "namespace", "name": namespace, "tools": []any{entry}}
	}
	body, err := marshalNoHTMLEscape(map[string]any{
		"model": model, "stream": stream, "input": "Print <the> & test marker.",
		"client_metadata": map[string]any{"x-codex-turn-metadata": map[string]string{"sandbox_mode": mode, "sandbox": "none"}},
		"tools":           []any{entry}, "unknown_field": json.Number("9007199254740993"),
	})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func checkFullAccessParameters(t *testing.T, parameters map[string]any, filtered bool) {
	t.Helper()
	properties := parameters["properties"].(map[string]any)
	for _, key := range []string{"sandbox_permissions", "justification", "prefix_rule"} {
		if _, exists := properties[key]; exists == filtered {
			t.Errorf("%s exists=%v, filtered=%v", key, exists, filtered)
		}
	}
	if !reflect.DeepEqual(parameters["required"], []any{"cmd"}) {
		t.Errorf("required=%v", parameters["required"])
	}
}

func TestResponsesFullAccessLocalConversion(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, mode := range []string{"danger-full-access", "workspace-write"} {
		for _, namespace := range []string{"", "functions", "application"} {
			for _, stream := range []bool{false, true} {
				t.Run(mode+"/"+namespace+map[bool]string{false: "/json", true: "/stream"}[stream], func(t *testing.T) {
					s := &Server{}
					router := gin.New()
					called := false
					router.POST("/v1/responses", s.responsesRequestMiddleware(), cloudPassthroughMiddleware("test"), middleware.ResponsesMiddleware(), func(c *gin.Context) {
						called = true
						var request api.ChatRequest
						if err := json.NewDecoder(c.Request.Body).Decode(&request); err != nil {
							t.Fatal(err)
						}
						if len(request.Tools) != 1 {
							t.Fatalf("tools=%d", len(request.Tools))
						}
						tool := request.Tools[0].Function
						wantName := "exec_command"
						if namespace != "" {
							wantName = namespace + "." + wantName
						}
						if tool.Name != wantName {
							t.Errorf("tool name=%s, want %s", tool.Name, wantName)
						}
						encoded, _ := json.Marshal(tool.Parameters)
						var parameters map[string]any
						if err := json.Unmarshal(encoded, &parameters); err != nil {
							t.Fatal(err)
						}
						checkFullAccessParameters(t, parameters, mode == "danger-full-access" && namespace != "application")
						// Model output is deliberately untouched: permission enforcement
						// remains in the client even when the schema was filtered.
						args := api.NewToolCallFunctionArguments()
						args.Set("cmd", "printf TEST_MARKER")
						args.Set("sandbox_permissions", "require_escalated")
						c.JSON(http.StatusOK, api.ChatResponse{
							Model:   request.Model,
							Done:    true,
							Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{ID: "call_test", Function: api.ToolCallFunction{Name: tool.Name, Arguments: args}}}},
						})
					})
					request := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(fullAccessRouteBody(t, "fixture", mode, namespace, stream)))
					request.Header.Set("Content-Type", "application/json")
					response := httptest.NewRecorder()
					router.ServeHTTP(response, request)
					if response.Code != http.StatusOK || !called {
						t.Fatalf("status=%d called=%v: %s", response.Code, called, response.Body.String())
					}
					for _, marker := range []string{"TEST_MARKER", "require_escalated", "call_test"} {
						if !strings.Contains(response.Body.String(), marker) {
							t.Errorf("response lost %q: %s", marker, response.Body.String())
						}
					}
				})
			}
		}
	}
}

func TestResponsesFullAccessCloudRoute(t *testing.T) {
	for _, mode := range []string{"danger-full-access", "workspace-write"} {
		for _, namespace := range []string{"", "functions", "application"} {
			t.Run(mode+"/"+namespace, func(t *testing.T) {
				t.Setenv("OLLAMA_NO_CLOUD", "false")
				const responseBody = `{"output":[{"type":"function_call","name":"exec_command","arguments":"{\"cmd\":\"printf TEST_MARKER\",\"sandbox_permissions\":\"require_escalated\"}"}]}`
				local, capture := newCompactionTestServer(t, func(_ int, w http.ResponseWriter, r *http.Request, _ []byte) {
					if r.Header.Get("Content-Encoding") != "" {
						t.Error("compressed body reached upstream")
					}
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, responseBody)
				})
				body := fullAccessRouteBody(t, "fixture:cloud", mode, namespace, false)
				var compressed bytes.Buffer
				writer, err := zstd.NewWriter(&compressed)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := writer.Write(body); err != nil {
					t.Fatal(err)
				}
				if err := writer.Close(); err != nil {
					t.Fatal(err)
				}
				request, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses", &compressed)
				if err != nil {
					t.Fatal(err)
				}
				request.Header.Set("Content-Type", "application/json")
				request.Header.Set("Content-Encoding", "zstd")
				response, err := local.Client().Do(request)
				if err != nil {
					t.Fatal(err)
				}
				defer response.Body.Close()
				got, _ := io.ReadAll(response.Body)
				if response.StatusCode != http.StatusOK || string(got) != responseBody {
					t.Fatalf("status=%d: %s", response.StatusCode, got)
				}
				paths, bodies := capture.snapshot()
				if len(paths) != 1 || paths[0] != "/v1/responses" {
					t.Fatalf("paths=%v", paths)
				}
				var sent map[string]any
				decoder := json.NewDecoder(bytes.NewReader(bodies[0]))
				decoder.UseNumber()
				if err := decoder.Decode(&sent); err != nil {
					t.Fatal(err)
				}
				if sent["model"] != "fixture" || sent["unknown_field"] != json.Number("9007199254740993") || sent["input"] != "Print <the> & test marker." {
					t.Fatalf("unrelated fields changed: %s", bodies[0])
				}
				if !bytes.Contains(bodies[0], []byte("Print <the> & test marker.")) || len(bodies[0]) > len(body) {
					t.Fatalf("forwarded request was HTML-escaped or inflated: before=%d after=%d", len(body), len(bodies[0]))
				}
				tool := sent["tools"].([]any)[0].(map[string]any)
				if namespace != "" {
					tool = tool["tools"].([]any)[0].(map[string]any)
				}
				checkFullAccessParameters(t, tool["parameters"].(map[string]any), mode == "danger-full-access" && namespace != "application")
			})
		}
	}
}
