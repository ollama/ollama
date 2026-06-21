package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

// kvEngine builds a minimal gin engine with only the KV slot routes wired to a
// zero-value Server. All the cases tested here return before scheduleRunner is
// reached, so no real runner is required.
func kvEngine() *gin.Engine {
	gin.SetMode(gin.TestMode)
	s := &Server{}
	r := gin.New()
	r.POST("/api/experimental/kv/save", s.KVSlotHandler("save"))
	r.POST("/api/experimental/kv/restore", s.KVSlotHandler("restore"))
	return r
}

func TestKVSlotHandlerRejects(t *testing.T) {
	tests := []struct {
		name        string
		slotSaveEnv string // value for OLLAMA_SLOT_SAVE_PATH ("" = feature disabled)
		body        string
		wantStatus  int
	}{
		{
			name:        "disabled when env unset",
			slotSaveEnv: "",
			body:        `{"model":"m","filename":"sys.bin"}`,
			wantStatus:  http.StatusBadRequest,
		},
		{
			name:        "missing model",
			slotSaveEnv: t.TempDir(),
			body:        `{"filename":"sys.bin"}`,
			wantStatus:  http.StatusBadRequest,
		},
		{
			name:        "missing filename",
			slotSaveEnv: t.TempDir(),
			body:        `{"model":"m"}`,
			wantStatus:  http.StatusBadRequest,
		},
		{
			name:        "filename with parent traversal",
			slotSaveEnv: t.TempDir(),
			body:        `{"model":"m","filename":"../escape.bin"}`,
			wantStatus:  http.StatusBadRequest,
		},
		{
			name:        "filename with subdir",
			slotSaveEnv: t.TempDir(),
			body:        `{"model":"m","filename":"sub/sys.bin"}`,
			wantStatus:  http.StatusBadRequest,
		},
		{
			name:        "filename with backslash subdir",
			slotSaveEnv: t.TempDir(),
			body:        `{"model":"m","filename":"sub\\sys.bin"}`,
			wantStatus:  http.StatusBadRequest,
		},
		{
			name:        "filename with leading slash",
			slotSaveEnv: t.TempDir(),
			body:        `{"model":"m","filename":"/etc/passwd"}`,
			wantStatus:  http.StatusBadRequest,
		},
		{
			name:        "filename with windows drive colon",
			slotSaveEnv: t.TempDir(),
			body:        `{"model":"m","filename":"C:sys.bin"}`,
			wantStatus:  http.StatusBadRequest,
		},
		{
			name:        "malformed json",
			slotSaveEnv: t.TempDir(),
			body:        `{`,
			wantStatus:  http.StatusBadRequest,
		},
	}

	for _, action := range []string{"save", "restore"} {
		for _, tt := range tests {
			t.Run(action+"/"+tt.name, func(t *testing.T) {
				t.Setenv("OLLAMA_SLOT_SAVE_PATH", tt.slotSaveEnv)
				r := kvEngine()

				w := httptest.NewRecorder()
				req := httptest.NewRequest(http.MethodPost, "/api/experimental/kv/"+action, strings.NewReader(tt.body))
				req.Header.Set("Content-Type", "application/json")
				r.ServeHTTP(w, req)

				if w.Code != tt.wantStatus {
					t.Fatalf("status = %d, want %d (body: %s)", w.Code, tt.wantStatus, w.Body.String())
				}
			})
		}
	}
}
