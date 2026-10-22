package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ollama/ollama/openai"
)

func TestNormalizeOllamaAgentMessagesPreservesConversation(t *testing.T) {
	body := []byte(`{"model":"test:cloud","input":[
  {"type":"message","role":"developer","content":"Environment"},
  {"type":"agent_message","id":"amsg_initial","author":"/root","recipient":"/root/child","content":[{"type":"input_text","text":"Read the "},{"type":"input_text","text":"fixture.\n"}],"internal_chat_message_metadata_passthrough":{"turn_id":"initial-turn"}},
  {"type":"function_call","call_id":"call_read","name":"read_file","arguments":"{}"},
  {"type":"function_call_output","call_id":"call_read","output":"fixture contents"},
  {"type":"agent_message","author":"/root/child","recipient":"/root","content":[{"type":"input_text","text":"Task finished."}]},
  {"type":"agent_message","author":"/root","recipient":"/root/child","content":[{"type":"input_text","text":"Now return RCA_FOLLOWUP."}]}
 ]}`)
	got, err := normalizeOllamaRequestBody(body, routingModel{})
	if err != nil {
		t.Fatal(err)
	}
	var request openai.ResponsesRequest
	if err := json.Unmarshal(got, &request); err != nil {
		t.Fatal(err)
	}
	chat, err := openai.FromResponsesRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	wantRoles := []string{"system", "user", "assistant", "tool", "user", "user"}
	wantContent := []string{"Environment", "Agent message from \"/root\" to \"/root/child\":\nRead the fixture.\n", "", "fixture contents", "Agent message from \"/root/child\" to \"/root\":\nTask finished.", "Agent message from \"/root\" to \"/root/child\":\nNow return RCA_FOLLOWUP."}
	if len(chat.Messages) != len(wantRoles) {
		t.Fatalf("got %d messages: %+v", len(chat.Messages), chat.Messages)
	}
	for i, msg := range chat.Messages {
		if msg.Role != wantRoles[i] || msg.Content != wantContent[i] {
			t.Errorf("message %d = %q %q; want %q %q", i, msg.Role, msg.Content, wantRoles[i], wantContent[i])
		}
	}
	if len(chat.Messages[2].ToolCalls) != 1 || chat.Messages[2].ToolCalls[0].ID != "call_read" || chat.Messages[3].ToolCallID != "call_read" {
		t.Fatal("tool call pairing changed")
	}
	var payload struct {
		Input []map[string]json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(got, &payload); err != nil {
		t.Fatal(err)
	}
	if string(payload.Input[1]["id"]) != `"amsg_initial"` || string(payload.Input[1]["internal_chat_message_metadata_passthrough"]) != `{"turn_id":"initial-turn"}` {
		t.Fatalf("message metadata changed: %s", got)
	}
	again, err := normalizeOllamaRequestBody(got, routingModel{})
	if err != nil || !bytes.Equal(got, again) {
		t.Fatalf("normalization is not idempotent: %s, %v", again, err)
	}
	native, changed, err := normalizeNativeRequestBody(body)
	if err != nil || changed || !bytes.Equal(native, body) {
		t.Fatalf("native conversation changed: %s, %v", native, err)
	}
}

func TestNormalizeOllamaAgentMessageRejectsIncompleteContent(t *testing.T) {
	for _, tt := range []struct{ name, content, want string }{
		{"unknown", `[{"type":"new_content","text":"secret-task"}]`, "unsupported Codex agent message content type"},
		{"missing type", `[{"text":"secret-task"}]`, "unsupported Codex agent message content type"},
		{"missing text", `[{"type":"input_text"}]`, "requires text"},
		{"null text", `[{"type":"input_text","text":null}]`, "requires text"},
		{"non-string text", `[{"type":"input_text","text":42}]`, "decode Codex agent message content"},
		{"empty", `[]`, "requires author, recipient, and content"},
		{"null", `null`, "requires author, recipient, and content"},
		{"not array", `"secret-task"`, "decode Codex agent message"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			item := []byte(fmt.Sprintf(`{"type":"agent_message","author":"/root","recipient":"/root/child","content":%s}`, tt.content))
			got, keep, err := normalizeOllamaInputItem(item)
			if err == nil || !strings.Contains(err.Error(), tt.want) || keep || got != nil {
				t.Fatalf("got %s, %v, %v", got, keep, err)
			}
			if strings.Contains(err.Error(), "secret-") {
				t.Fatalf("content exposed in error: %v", err)
			}
		})
	}
	for _, field := range []string{"author", "recipient"} {
		t.Run("missing "+field, func(t *testing.T) {
			item := map[string]any{"type": "agent_message", "author": "/root", "recipient": "/root/child", "content": []any{map[string]string{"type": "input_text", "text": "task"}}}
			delete(item, field)
			raw, _ := json.Marshal(item)
			if _, _, err := normalizeOllamaInputItem(raw); err == nil {
				t.Fatalf("accepted message without %s", field)
			}
		})
	}
}

func TestNormalizeOllamaAgentMessageAcceptsEncryptedContentAsText(t *testing.T) {
	item := []byte(`{"type":"agent_message","author":"/root","recipient":"/root/child","content":[{"type":"input_text","text":"Payload:\n"},{"type":"encrypted_content","encrypted_content":"secret-task"},{"type":"input_text","text":"trailing instruction"}]}`)
	got, keep, err := normalizeOllamaInputItem(item)
	if err != nil || !keep {
		t.Fatalf("normalize failed: %v", err)
	}
	var msg struct {
		Type    string `json:"type"`
		Role    string `json:"role"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(got, &msg); err != nil {
		t.Fatal(err)
	}
	if msg.Type != "message" || msg.Role != "user" {
		t.Fatalf("converted message = %q %q", msg.Type, msg.Role)
	}
	want := []struct{ Type, Text string }{
		{"input_text", "Agent message from \"/root\" to \"/root/child\":\n"},
		{"input_text", "Payload:\n"},
		{"input_text", "secret-task"},
		{"input_text", "trailing instruction"},
	}
	if len(msg.Content) != len(want) {
		t.Fatalf("content parts = %+v", msg.Content)
	}
	for i, part := range msg.Content {
		if part.Type != want[i].Type || part.Text != want[i].Text {
			t.Errorf("part %d = %q %q; want %q %q", i, part.Type, part.Text, want[i].Type, want[i].Text)
		}
	}
}

func TestCodexDesktopEncryptedAgentMessageReachesOllamaAsText(t *testing.T) {
	var bodies [][]byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer upstream.Close()
	h := newTestCodexDesktop(t, upstream.URL, upstream.URL, writeCatalog(t, "glm-5.3-flash:cloud", "glm-5.3:cloud"))
	for _, model := range []string{"glm-5.3:cloud", "gpt-5.6-terra"} {
		t.Run(model, func(t *testing.T) {
			body := fmt.Sprintf(`{"model":%q,"input":[{"type":"message","role":"user","content":"old task"},{"type":"agent_message","author":"/root","recipient":"/root/child","content":[{"type":"input_text","text":"Payload:"},{"type":"encrypted_content","encrypted_content":"secret-cipher"}]}]}`, model)
			req := httptest.NewRequest("POST", CodexDesktopPathPrefix+"/v1/responses", strings.NewReader(body))
			req.RemoteAddr = "127.0.0.1:1234"
			req.Header.Set("Authorization", "Bearer native-test")
			req.Header.Set("ChatGPT-Account-ID", "test-account")
			w := httptest.NewRecorder()
			h.ServeHTTP(w, req)
			if w.Code != 204 {
				t.Fatalf("request = %d: %s", w.Code, w.Body)
			}
			forwarded := bodies[len(bodies)-1]
			if !bytes.Contains(forwarded, []byte("secret-cipher")) {
				t.Errorf("payload lost: %s", forwarded)
			}
			if model == "gpt-5.6-terra" {
				if !bytes.Contains(forwarded, []byte(`"type":"agent_message"`)) {
					t.Errorf("native transcript changed: %s", forwarded)
				}
				return
			}
			if bytes.Contains(forwarded, []byte(`"type":"agent_message"`)) {
				t.Errorf("agent message not converted for Ollama: %s", forwarded)
			}
			if !bytes.Contains(forwarded, []byte(`\"type\":\"input_text\",\"text\":\"secret-cipher\"`)) &&
				!bytes.Contains(forwarded, []byte(`"text":"secret-cipher"`)) {
				t.Errorf("encrypted payload not flattened to text: %s", forwarded)
			}
		})
	}
}

func TestAgentMessageEnvelopeMatchesOpenAI(t *testing.T) {
	item := []byte(`{"type":"agent_message","author":"/root","recipient":"/root/child","content":[{"type":"input_text","text":"task"}]}`)
	got, keep, err := normalizeOllamaInputItem(item)
	if err != nil || !keep {
		t.Fatalf("normalize failed: %v", err)
	}
	var msg struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(got, &msg); err != nil {
		t.Fatal(err)
	}
	if len(msg.Content) == 0 {
		t.Fatal("no content parts")
	}
	want := fmt.Sprintf(openai.AgentMessageEnvelopeFormat, "/root", "/root/child")
	if msg.Content[0].Text != want {
		t.Fatalf("envelope = %q; want %q (keep the proxy envelope in sync with openai.AgentMessageEnvelopeFormat)", msg.Content[0].Text, want)
	}
}
