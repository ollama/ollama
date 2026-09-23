//go:build windows || darwin

package ui

import (
	"database/sql"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/app/history"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/ui/responses"
)

func TestReadOnlyChatExportAndDeletion(t *testing.T) {
	st := &store.Store{DBPath: filepath.Join(t.TempDir(), "app.sqlite")}
	defer st.Close()
	if _, err := st.Settings(); err != nil {
		t.Fatal(err)
	}
	db, err := sql.Open("sqlite3", st.DBPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`INSERT INTO chats (id, title) VALUES ('saved-chat', 'Keep this chat');
		INSERT INTO messages (chat_id, role, content, model_name) VALUES ('saved-chat', 'assistant', 'Original message', 'gpt-oss:120b-cloud');
		INSERT INTO attachments (message_id, filename, data) VALUES (last_insert_rowid(), 'notes.txt', X'4142');`); err != nil {
		t.Fatal(err)
	}
	calls := 0
	s := &Server{Store: st, Token: "test-token", Restart: func() {}, ExportChat: func(chat store.Chat) (*history.Result, error) {
		calls++
		if chat.Title != "Keep this chat" || len(chat.Messages) != 1 || string(chat.Messages[0].Attachments[0].Data) != "AB" || chat.Messages[0].Model != "gpt-oss:120b-cloud" {
			t.Fatal("export did not receive the full saved conversation")
		}
		return history.Export(chat, t.TempDir())
	}}
	s.ExportAllChats = func() (*history.Result, error) {
		calls++
		return &history.Result{Path: "chats.zip"}, nil
	}
	handler := s.Handler()
	request := func(method, path, token string) *httptest.ResponseRecorder {
		r := httptest.NewRequest(method, path, strings.NewReader(`{}`))
		r.AddCookie(&http.Cookie{Name: "token", Value: token})
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, r)
		return w
	}
	for _, path := range []string{"/connect", "/settings", "/c/saved-chat", "/api/v1/chats", "/api/v1/chat/saved-chat"} {
		if w := request("GET", path, s.Token); w.Code != http.StatusOK {
			t.Fatalf("%s: %d %s", path, w.Code, w.Body.String())
		}
	}
	for _, retired := range []struct{ method, path string }{
		{"POST", "/api/v1/chat/saved-chat"}, {"POST", "/api/v1/create-chat"}, {"PUT", "/api/v1/chat/saved-chat/rename"},
	} {
		if w := request(retired.method, retired.path, s.Token); w.Code != http.StatusMethodNotAllowed {
			t.Fatalf("retired chat action remains available: %+v: %d", retired, w.Code)
		}
	}
	for _, action := range []struct{ method, path string }{{"POST", "/api/v1/chat/saved-chat/export"}, {"POST", "/api/v1/chats/export"}, {"DELETE", "/api/v1/chat/saved-chat"}} {
		if w := request(action.method, action.path, ""); w.Code != http.StatusForbidden {
			t.Fatalf("unauthenticated action allowed: %+v", action)
		}
	}
	if calls != 0 {
		t.Fatal("unauthenticated export opened the folder picker")
	}
	if w := request("POST", "/api/v1/chat/saved-chat/export", s.Token); w.Code != http.StatusOK || calls != 1 || !strings.Contains(w.Body.String(), "path") {
		t.Fatalf("export failed: %d %s", w.Code, w.Body.String())
	}
	if w := request("POST", "/api/v1/chats/export", s.Token); w.Code != http.StatusOK || calls != 2 || !strings.Contains(w.Body.String(), "chats.zip") {
		t.Fatalf("bulk export failed: %d %s", w.Code, w.Body.String())
	}
	for _, path := range []string{"/api/v1/chat/saved-chat/export", "/api/v1/chats/export"} {
		s.ExportChat = func(store.Chat) (*history.Result, error) { return nil, nil }
		s.ExportAllChats = func() (*history.Result, error) { return nil, nil }
		if w := request("POST", path, s.Token); w.Code != http.StatusOK || strings.TrimSpace(w.Body.String()) != "null" {
			t.Fatal("cancellation should not be an error")
		}
		s.ExportChat = func(store.Chat) (*history.Result, error) { return nil, errors.New("disk is full") }
		s.ExportAllChats = func() (*history.Result, error) { return nil, errors.New("disk is full") }
		if w := request("POST", path, s.Token); w.Code != http.StatusInternalServerError || !strings.Contains(w.Body.String(), "disk is full") {
			t.Fatal("export failure should be visible")
		}
	}
	if chat, err := st.Chat("saved-chat"); err != nil || chat.Title != "Keep this chat" || chat.Messages[0].Content != "Original message" {
		t.Fatal("export or retired actions changed the saved chat")
	}
	if w := request("DELETE", "/api/v1/chat/saved-chat", s.Token); w.Code != http.StatusOK {
		t.Fatalf("deletion failed: %d %s", w.Code, w.Body.String())
	}
	if _, err := st.Chat("saved-chat"); err == nil {
		t.Fatal("deleted chat remains in the store")
	}

	latestState := `{"page_stack":["https://example.com/latest"]}`
	if _, err := db.Exec(`INSERT INTO chats (id) VALUES ('legacy');
		INSERT INTO messages (chat_id, role, tool_result) VALUES ('legacy', 'tool', '{"page_stack":["https://example.com/older"]}');
		INSERT INTO messages (chat_id, role, tool_result) VALUES ('legacy', 'tool', ?);
		INSERT INTO messages (chat_id, role, tool_result) VALUES ('legacy', 'tool', '{"answer":"unrelated tool result"}');`, latestState); err != nil {
		t.Fatal(err)
	}
	for _, storedState := range []string{"", `{"page_stack":["https://example.com/saved"]}`} {
		if _, err := db.Exec(`UPDATE chats SET browser_state = ? WHERE id = 'legacy'`, storedState); err != nil {
			t.Fatal(err)
		}
		w := request("GET", "/api/v1/chat/legacy", s.Token)
		var data responses.ChatResponse
		if err := json.Unmarshal(w.Body.Bytes(), &data); err != nil || w.Code != http.StatusOK {
			t.Fatalf("read legacy chat: %d %s, %v", w.Code, w.Body.String(), err)
		}
		want := storedState
		if want == "" {
			want = latestState
		}
		if string(data.Chat.BrowserState) != want {
			t.Fatalf("wrong citation state: got %s, want %s", data.Chat.BrowserState, want)
		}
		chat, err := st.Chat("legacy")
		if err != nil || string(chat.BrowserState) != storedState {
			t.Fatalf("reading citations changed saved history: %v", err)
		}
	}
}
