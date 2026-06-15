//go:build integration

package integration

// slot_test.go exercises the /api/slot/{save,restore,erase} endpoints round-
// trip against a real llama-server subprocess. Requires the server to be
// started with OLLAMA_SLOT_SAVE_PATH set to a writable directory; this test
// sets that env before InitServerConnection spawns the server.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestSlotSaveRestoreRoundTrip(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("OLLAMA_SLOT_SAVE_PATH", dir)

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()

	client, endpoint, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	pullOrSkip(ctx, t, client, smol)

	// Prime the slot by running a short completion. The state we save is
	// whatever KV the runner has after this turn.
	stream := false
	if err := client.Generate(ctx, &api.GenerateRequest{
		Model:  smol,
		Prompt: "The capital of France is",
		Stream: &stream,
		Options: map[string]any{
			"num_predict": 4,
			"seed":        7,
		},
	}, func(api.GenerateResponse) error { return nil }); err != nil {
		t.Fatalf("prime generate: %v", err)
	}

	saveBody := map[string]any{"model": smol, "id_slot": 0, "filename": "rt.bin"}
	doSlot(t, endpoint, "save", saveBody, http.StatusOK)

	if _, err := os.Stat(filepath.Join(dir, "rt.bin")); err != nil {
		t.Fatalf("save file not on disk: %v", err)
	}

	// Erase the in-memory slot, then restore from disk. A successful restore
	// must report a non-zero n_restored token count.
	doSlot(t, endpoint, "erase", map[string]any{"model": smol, "id_slot": 0}, http.StatusOK)

	raw := doSlot(t, endpoint, "restore", saveBody, http.StatusOK)
	var rest struct {
		NRestored int `json:"n_restored"`
	}
	if err := json.Unmarshal(raw, &rest); err != nil {
		t.Fatalf("decode restore reply: %v (raw=%s)", err, raw)
	}
	if rest.NRestored <= 0 {
		t.Fatalf("expected n_restored>0, got %d (raw=%s)", rest.NRestored, raw)
	}
}

// TestSlotDisabledWhenEnvUnset confirms that without OLLAMA_SLOT_SAVE_PATH the
// endpoint cleanly reports 501 rather than crashing the runner or returning a
// confusing scheduler error.
func TestSlotDisabledWhenEnvUnset(t *testing.T) {
	t.Setenv("OLLAMA_SLOT_SAVE_PATH", "")

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()

	_, endpoint, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	doSlot(t, endpoint, "save",
		map[string]any{"model": smol, "id_slot": 0, "filename": "x.bin"},
		http.StatusNotImplemented)
}

func doSlot(t *testing.T, endpoint, action string, body map[string]any, wantCode int) []byte {
	t.Helper()

	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}

	url := fmt.Sprintf("%s/api/slot/%s", endpoint, action)
	req, err := http.NewRequest("POST", url, bytes.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("%s: %v", action, err)
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != wantCode {
		t.Fatalf("%s: status want %d got %d body=%s", action, wantCode, resp.StatusCode, raw)
	}
	return raw
}
