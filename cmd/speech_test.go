package cmd

import (
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestSynthesize(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct{ Input string }
		if r.URL.Path != "/v1/audio/speech" || json.NewDecoder(r.Body).Decode(&req) != nil || req.Input == "" {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error":{"message":"input is required"}}`))
			return
		}
		header := make([]byte, 44)
		binary.LittleEndian.PutUint32(header[4:], 0xffffffff)
		binary.LittleEndian.PutUint32(header[40:], 0xffffffff)
		w.Write(append(header, 1, 0, 2, 0))
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	wav, err := synthesize(t.Context(), "phonon", "hello")
	if err != nil || binary.LittleEndian.Uint32(wav[4:]) != 40 || binary.LittleEndian.Uint32(wav[40:]) != 4 {
		t.Fatalf("WAV lengths not recorded: %x, %v", wav, err)
	}
	if _, err := synthesize(t.Context(), "phonon", ""); err == nil || err.Error() != "input is required" {
		t.Fatalf("server error lost: %v", err)
	}
}
