package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

// TestSlotHandlersDisabledByDefault verifies that without OLLAMA_SLOT_SAVE_PATH
// the slot endpoints respond 501 NOT_IMPLEMENTED before any model scheduling
// happens — so a stock Ollama install does not silently forward to a runner.
func TestSlotHandlersDisabledByDefault(t *testing.T) {
	t.Setenv("OLLAMA_SLOT_SAVE_PATH", "")

	gin.SetMode(gin.TestMode)

	cases := []struct {
		path    string
		body    string
		handler gin.HandlerFunc
	}{
		{"/api/slot/save", `{"model":"m","id_slot":0,"filename":"f.bin"}`, (&Server{}).SlotSaveHandler},
		{"/api/slot/restore", `{"model":"m","id_slot":0,"filename":"f.bin"}`, (&Server{}).SlotRestoreHandler},
		{"/api/slot/erase", `{"model":"m","id_slot":0}`, (&Server{}).SlotEraseHandler},
	}

	for _, tc := range cases {
		t.Run(tc.path, func(t *testing.T) {
			r := gin.New()
			r.POST(tc.path, tc.handler)

			w := httptest.NewRecorder()
			req, _ := http.NewRequest("POST", tc.path, strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")
			r.ServeHTTP(w, req)

			if w.Code != http.StatusNotImplemented {
				t.Fatalf("expected 501 when slot path is unset, got %d: %s", w.Code, w.Body.String())
			}
			if !strings.Contains(w.Body.String(), "OLLAMA_SLOT_SAVE_PATH") {
				t.Fatalf("expected error to mention OLLAMA_SLOT_SAVE_PATH, got %s", w.Body.String())
			}
		})
	}
}

// TestSlotSaveRejectsBadInput validates request-shape errors return 400, not 5xx,
// and never reach the scheduler.
func TestSlotSaveRejectsBadInput(t *testing.T) {
	t.Setenv("OLLAMA_SLOT_SAVE_PATH", "/tmp/ollama-slots-test")

	gin.SetMode(gin.TestMode)

	cases := []struct {
		name string
		body string
		want int
	}{
		{"missing-model", `{"id_slot":0,"filename":"f.bin"}`, http.StatusBadRequest},
		{"missing-filename", `{"model":"m","id_slot":0}`, http.StatusBadRequest},
		{"negative-id-slot", `{"model":"m","id_slot":-1,"filename":"f.bin"}`, http.StatusBadRequest},
		{"not-json", `not json`, http.StatusBadRequest},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := gin.New()
			r.POST("/api/slot/save", (&Server{}).SlotSaveHandler)

			w := httptest.NewRecorder()
			req, _ := http.NewRequest("POST", "/api/slot/save", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")
			r.ServeHTTP(w, req)

			if w.Code != tc.want {
				t.Fatalf("status: want %d got %d body=%s", tc.want, w.Code, w.Body.String())
			}
		})
	}
}

// TestSlotResponseDecode verifies that the SlotResponse shape round-trips a
// representative upstream llama-server reply, including the optional sidecar
// fields that LuminaNAO's LSCKPT2 build adds.
func TestSlotResponseDecode(t *testing.T) {
	upstream := []byte(`{
		"id_slot": 2,
		"filename": "session-a.bin",
		"n_saved": 1024,
		"n_written": 8388608,
		"timings": {"save_ms": 12.5}
	}`)

	var resp SlotResponse
	if err := json.Unmarshal(upstream, &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.IDSlot != 2 || resp.Filename != "session-a.bin" {
		t.Fatalf("shape mismatch: %+v", resp)
	}
	if resp.NSaved != 1024 || resp.NWritten != 8388608 {
		t.Fatalf("counters: %+v", resp)
	}

	// Re-encode and confirm timings survive as raw JSON.
	out, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if !bytes.Contains(out, []byte(`"save_ms":12.5`)) {
		t.Fatalf("timings dropped: %s", out)
	}
}
