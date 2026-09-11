package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"golang.org/x/sync/semaphore"
)

func mustExist(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
}

func mustNotExist(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("stat %s: got %v, want %v", path, err, os.ErrNotExist)
	}
}

func mustReadFile(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(data)
}

func mustWriteFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// serverPort returns the port httptest bound, which the runner needs because
// it addresses llama-server by port rather than by URL.
func serverPort(t *testing.T, server *httptest.Server) int {
	t.Helper()
	parsed, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("parse %s: %v", server.URL, err)
	}
	_, portText, err := net.SplitHostPort(parsed.Host)
	if err != nil {
		t.Fatalf("split %s: %v", parsed.Host, err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatalf("parse port %s: %v", portText, err)
	}
	return port
}

func TestLlamaPrefillCacheSaveRestore(t *testing.T) {
	cachePath := t.TempDir()
	var mu sync.Mutex
	var actions []string

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Filename string `json:"filename"`
		}
		// t.Fatalf must not run outside the test goroutine, so handler
		// failures are reported and answered with an error status.
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode slot request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		action := r.URL.Query().Get("action")
		id := strings.TrimPrefix(r.URL.Path, "/slots/")
		mu.Lock()
		actions = append(actions, action+":"+id+":"+body.Filename)
		mu.Unlock()
		if action == "save" {
			if err := os.WriteFile(filepath.Join(cachePath, body.Filename), []byte("slot "+id), 0o600); err != nil {
				t.Errorf("write %s: %v", body.Filename, err)
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
		}
		w.Header().Set("Content-Type", "application/json")
		if action == "save" && id == "1" {
			w.WriteHeader(http.StatusNotImplemented)
			_, _ = w.Write([]byte(`{"error":"media slot"}`))
			return
		}
		if id == "0" {
			_, _ = w.Write([]byte(`{"n_saved":3}`))
		} else {
			_, _ = w.Write([]byte(`{"n_saved":0}`))
		}
	})
	server := httptest.NewServer(handler)
	defer server.Close()

	runner := &llamaServerRunner{
		port:   serverPort(t, server),
		client: server.Client(),
		sem:    semaphore.NewWeighted(2),
		launch: llamaServerLaunchConfig{
			numParallel: 2,
			config:      LlamaServerConfig{PrefillCachePath: cachePath},
		},
	}

	// An unsupported slot must not retain its previous snapshot.
	mustWriteFile(t, filepath.Join(cachePath, "slot-1.bin"), "stale")

	if err := runner.SavePrefillCache(t.Context()); err != nil {
		t.Fatalf("SavePrefillCache() = %v, want nil", err)
	}
	mustExist(t, filepath.Join(cachePath, "slot-0.bin"))
	mustNotExist(t, filepath.Join(cachePath, "slot-1.bin"))
	mustExist(t, filepath.Join(cachePath, "manifest.json"))
	mustNotExist(t, filepath.Join(cachePath, "slot-0.bin.tmp"))
	mustNotExist(t, filepath.Join(cachePath, "slot-1.bin.tmp"))

	// Replacing an existing published checkpoint must also work on Windows.
	if err := runner.SavePrefillCache(t.Context()); err != nil {
		t.Fatalf("second SavePrefillCache() = %v, want nil", err)
	}
	mustExist(t, filepath.Join(cachePath, "manifest.json"))

	mu.Lock()
	actions = nil
	mu.Unlock()
	if err := runner.RestorePrefillCache(t.Context()); err != nil {
		t.Fatalf("RestorePrefillCache() = %v, want nil", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if want := []string{"restore:0:slot-0.bin"}; !slices.Equal(actions, want) {
		t.Fatalf("slot actions = %v, want %v", actions, want)
	}
}

func TestLlamaPrefillCacheSkipsEmptySlot(t *testing.T) {
	cachePath := t.TempDir()
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Filename string `json:"filename"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode slot request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if err := os.WriteFile(filepath.Join(cachePath, body.Filename), []byte("empty slot"), 0o600); err != nil {
			t.Errorf("write %s: %v", body.Filename, err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"n_saved":0}`))
	})
	server := httptest.NewServer(handler)
	defer server.Close()

	runner := &llamaServerRunner{
		port:   serverPort(t, server),
		client: server.Client(),
		sem:    semaphore.NewWeighted(1),
		launch: llamaServerLaunchConfig{
			numParallel: 1,
			config:      LlamaServerConfig{PrefillCachePath: cachePath},
		},
	}

	mustWriteFile(t, filepath.Join(cachePath, "slot-0.bin"), "stale")
	if err := runner.SavePrefillCache(t.Context()); err != nil {
		t.Fatalf("SavePrefillCache() = %v, want nil", err)
	}
	mustNotExist(t, filepath.Join(cachePath, "slot-0.bin.tmp"))
	mustNotExist(t, filepath.Join(cachePath, "slot-0.bin"))

	data, err := os.ReadFile(filepath.Join(cachePath, "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest llamaPrefillCacheManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if len(manifest.Slots) != 0 {
		t.Fatalf("manifest = %+v, want no saved slots", manifest)
	}
}

// manifestForSlots builds the manifest a save would have written for the slot
// files already in cachePath. A named slot with no file gets a zero checksum,
// which is what the missing-slot case needs.
func manifestForSlots(t *testing.T, cachePath string, ids ...int) string {
	t.Helper()
	slots := make([]llamaPrefillCacheSlot, 0, len(ids))
	for _, id := range ids {
		crc, _, err := checksumPrefillCacheSlot(filepath.Join(cachePath, fmt.Sprintf("slot-%d.bin", id)))
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("checksum slot %d: %v", id, err)
		}
		slots = append(slots, llamaPrefillCacheSlot{ID: id, CRC: crc})
	}
	data, err := json.Marshal(llamaPrefillCacheManifest{Slots: slots})
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	return string(data)
}

func TestLlamaPrefillCacheErasesSlotOnFailedRestore(t *testing.T) {
	// llama-server answers 400 both for a bad save file and for runtime
	// failures such as insufficient KV space, so no status proves the
	// snapshot is corrupt: every failed restore erases the slot and keeps
	// the snapshot for the next load.
	for name, status := range map[string]int{
		"invalid request": http.StatusBadRequest,
		"unavailable":     http.StatusServiceUnavailable,
	} {
		t.Run(name, func(t *testing.T) {
			cachePath := t.TempDir()
			mustWriteFile(t, filepath.Join(cachePath, "slot-0.bin"), "kv")
			mustWriteFile(t, filepath.Join(cachePath, "manifest.json"), manifestForSlots(t, cachePath, 0))

			var mu sync.Mutex
			var actions []string
			handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				action := r.URL.Query().Get("action")
				id := strings.TrimPrefix(r.URL.Path, "/slots/")
				mu.Lock()
				actions = append(actions, action+":"+id)
				mu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				if action == "restore" {
					w.WriteHeader(status)
					_, _ = w.Write([]byte(`{"error":"cannot restore"}`))
					return
				}
				_, _ = w.Write([]byte(`{}`))
			})
			server := httptest.NewServer(handler)
			defer server.Close()

			runner := &llamaServerRunner{
				port:   serverPort(t, server),
				client: server.Client(),
				sem:    semaphore.NewWeighted(1),
				launch: llamaServerLaunchConfig{
					numParallel: 1,
					config:      LlamaServerConfig{PrefillCachePath: cachePath},
				},
			}

			if err := runner.RestorePrefillCache(t.Context()); err == nil {
				t.Fatal("RestorePrefillCache() = nil, want an error")
			}
			// The slot must be erased so a partial restore cannot corrupt
			// later completions.
			mu.Lock()
			if want := []string{"restore:0", "erase:0"}; !slices.Equal(actions, want) {
				mu.Unlock()
				t.Fatalf("slot actions = %v, want %v", actions, want)
			}
			mu.Unlock()
			mustExist(t, filepath.Join(cachePath, "manifest.json"))
			mustExist(t, filepath.Join(cachePath, "slot-0.bin"))
		})
	}
}

func TestLlamaPrefillCacheSaveRecreatesMissingDirectory(t *testing.T) {
	// An unusable snapshot is deleted while the runner keeps serving, so the
	// next save has to rebuild the directory llama-server was launched with.
	cachePath := filepath.Join(t.TempDir(), "cache")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"n_saved":0}`))
	}))
	defer server.Close()
	runner := &llamaServerRunner{
		port:   serverPort(t, server),
		client: server.Client(),
		sem:    semaphore.NewWeighted(1),
		launch: llamaServerLaunchConfig{
			numParallel: 1,
			config:      LlamaServerConfig{PrefillCachePath: cachePath},
		},
	}
	if err := runner.SavePrefillCache(t.Context()); err != nil {
		t.Fatalf("SavePrefillCache() = %v, want nil", err)
	}
	mustExist(t, filepath.Join(cachePath, "manifest.json"))
}

func TestLlamaPrefillCacheKeepsPreviousManifestOnFailedSave(t *testing.T) {
	cachePath := t.TempDir()
	mustWriteFile(t, filepath.Join(cachePath, "slot-0.bin"), "kv")
	manifest := manifestForSlots(t, cachePath, 0)
	mustWriteFile(t, filepath.Join(cachePath, "manifest.json"), manifest)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()
	runner := &llamaServerRunner{
		port:   serverPort(t, server),
		client: server.Client(),
		sem:    semaphore.NewWeighted(1),
		launch: llamaServerLaunchConfig{
			numParallel: 1,
			config:      LlamaServerConfig{PrefillCachePath: cachePath},
		},
	}
	if err := runner.SavePrefillCache(t.Context()); err == nil {
		t.Fatal("SavePrefillCache() = nil, want an error")
	}
	// A save that fails before replacing anything must leave the last good
	// generation restorable rather than invalidating it up front.
	if got := mustReadFile(t, filepath.Join(cachePath, "manifest.json")); got != manifest {
		t.Fatalf("manifest.json = %q, want the previous generation %q", got, manifest)
	}
	if got := mustReadFile(t, filepath.Join(cachePath, "slot-0.bin")); got != "kv" {
		t.Fatalf("slot-0.bin = %q, want %q", got, "kv")
	}
}

func TestLlamaPrefillCacheMissingSlotFailsOpen(t *testing.T) {
	cachePath := t.TempDir()
	mustWriteFile(t, filepath.Join(cachePath, "slot-0.bin"), "kv0")
	mustWriteFile(t, filepath.Join(cachePath, "manifest.json"), manifestForSlots(t, cachePath, 0, 1))

	var actions atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		actions.Add(1)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()
	runner := &llamaServerRunner{
		port:   serverPort(t, server),
		client: server.Client(),
		sem:    semaphore.NewWeighted(2),
		launch: llamaServerLaunchConfig{
			numParallel: 2,
			config:      LlamaServerConfig{PrefillCachePath: cachePath},
		},
	}
	err := runner.RestorePrefillCache(t.Context())
	if err == nil || !strings.Contains(err.Error(), "slot 1 is missing") {
		t.Fatalf("RestorePrefillCache() = %v, want a missing slot error", err)
	}
	if got := actions.Load(); got != 0 {
		t.Fatalf("slot actions = %d, want 0: every slot must be accounted for before restore", got)
	}
	mustNotExist(t, cachePath)
}

func TestLlamaPrefillCacheCanceledSaveCreatesNothing(t *testing.T) {
	cachePath := filepath.Join(t.TempDir(), "cache")
	runner := &llamaServerRunner{
		sem: semaphore.NewWeighted(1),
		launch: llamaServerLaunchConfig{
			numParallel: 1,
			config:      LlamaServerConfig{PrefillCachePath: cachePath},
		},
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	// Shutdown cancels in-flight saves while it removes the cache root, so a
	// canceled save that still created its directory would leak it.
	if err := runner.SavePrefillCache(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("SavePrefillCache() = %v, want %v", err, context.Canceled)
	}
	mustNotExist(t, cachePath)
}

func TestLlamaPrefillCacheKeepsSnapshotOnInterruptedRestore(t *testing.T) {
	cachePath := t.TempDir()
	mustWriteFile(t, filepath.Join(cachePath, "slot-0.bin"), "kv")
	mustWriteFile(t, filepath.Join(cachePath, "manifest.json"), manifestForSlots(t, cachePath, 0))

	runner := &llamaServerRunner{
		sem: semaphore.NewWeighted(1),
		launch: llamaServerLaunchConfig{
			numParallel: 1,
			config:      LlamaServerConfig{PrefillCachePath: cachePath},
		},
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := runner.RestorePrefillCache(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("RestorePrefillCache() = %v, want %v", err, context.Canceled)
	}
	mustExist(t, filepath.Join(cachePath, "manifest.json"))
	mustExist(t, filepath.Join(cachePath, "slot-0.bin"))
}

func TestLlamaPrefillCacheRejectsInvalidManifest(t *testing.T) {
	// A manifest that cannot describe this runner's slots is rejected before
	// the server is touched, and is deleted rather than retried so the next
	// load is a clean cold prefill instead of the same failure forever.
	crc := crc32.Checksum([]byte("kv"), llamaPrefillCacheCRC)
	for name, manifest := range map[string]string{
		"slot out of range": fmt.Sprintf(`{"slots":[{"id":0,"crc32c":%d},{"id":1,"crc32c":%d}]}`, crc, crc),
		"duplicate slot":    fmt.Sprintf(`{"slots":[{"id":0,"crc32c":%d},{"id":0,"crc32c":%d}]}`, crc, crc),
		"checksum mismatch": fmt.Sprintf(`{"slots":[{"id":0,"crc32c":%d}]}`, crc+1),
		"malformed json":    `{"slots":`,
	} {
		t.Run(name, func(t *testing.T) {
			cachePath := t.TempDir()
			mustWriteFile(t, filepath.Join(cachePath, "manifest.json"), manifest)
			// Every slot the manifests name has a file, so the manifest itself
			// is the only thing that can make the snapshot unusable.
			mustWriteFile(t, filepath.Join(cachePath, "slot-0.bin"), "kv")
			mustWriteFile(t, filepath.Join(cachePath, "slot-1.bin"), "kv")

			var actions atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				actions.Add(1)
				_, _ = w.Write([]byte(`{}`))
			}))
			defer server.Close()
			runner := &llamaServerRunner{
				port:   serverPort(t, server),
				client: server.Client(),
				sem:    semaphore.NewWeighted(1),
				launch: llamaServerLaunchConfig{
					numParallel: 1,
					config:      LlamaServerConfig{PrefillCachePath: cachePath},
				},
			}
			if err := runner.RestorePrefillCache(t.Context()); err == nil {
				t.Fatal("RestorePrefillCache() = nil, want an error")
			}
			if got := actions.Load(); got != 0 {
				t.Fatalf("slot actions = %d, want 0", got)
			}
			mustNotExist(t, cachePath)
		})
	}
}

func TestLlamaPrefillCacheWithoutManifestIsColdMiss(t *testing.T) {
	cachePath := t.TempDir()
	var actions atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		actions.Add(1)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()
	runner := &llamaServerRunner{
		port:   serverPort(t, server),
		client: server.Client(),
		sem:    semaphore.NewWeighted(1),
		launch: llamaServerLaunchConfig{
			numParallel: 1,
			config:      LlamaServerConfig{PrefillCachePath: cachePath},
		},
	}
	// Every runner starts without a manifest, so an absent one is the normal
	// first load rather than a failure.
	if err := runner.RestorePrefillCache(t.Context()); err != nil {
		t.Fatalf("RestorePrefillCache() = %v, want nil", err)
	}
	if got := actions.Load(); got != 0 {
		t.Fatalf("slot actions = %d, want 0", got)
	}
	mustExist(t, cachePath)
}

func TestLlamaPrefillCacheDisabledIsNoOp(t *testing.T) {
	// With persistence off the configured path is empty. Both directions must
	// return having touched nothing: without the guard, restore would read a
	// relative manifest.json out of the daemon's working directory and save
	// would report a failure on every unload.
	dir := t.TempDir()
	mustWriteFile(t, filepath.Join(dir, "slot-0.bin"), "kv")
	mustWriteFile(t, filepath.Join(dir, "manifest.json"), manifestForSlots(t, dir, 0))
	t.Chdir(dir)

	var actions atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		actions.Add(1)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()
	runner := &llamaServerRunner{
		port:   serverPort(t, server),
		client: server.Client(),
		sem:    semaphore.NewWeighted(1),
		launch: llamaServerLaunchConfig{numParallel: 1},
	}
	if err := runner.SavePrefillCache(t.Context()); err != nil {
		t.Fatalf("SavePrefillCache() = %v, want nil", err)
	}
	if err := runner.RestorePrefillCache(t.Context()); err != nil {
		t.Fatalf("RestorePrefillCache() = %v, want nil", err)
	}
	if got := actions.Load(); got != 0 {
		t.Fatalf("slot actions = %d, want 0", got)
	}
}

func TestAppendPrefillCacheArgsFailOpen(t *testing.T) {
	unchanged := []string{"--model", "m"}
	if got := appendPrefillCacheArgs(unchanged, ""); !slices.Equal(got, unchanged) {
		t.Fatalf("appendPrefillCacheArgs(%v, \"\") = %v, want %v", unchanged, got, unchanged)
	}

	cachePath := filepath.Join(t.TempDir(), "cache")
	want := []string{"--slot-save-path", filepath.Clean(cachePath) + string(os.PathSeparator)}
	if got := appendPrefillCacheArgs(nil, cachePath); !slices.Equal(got, want) {
		t.Fatalf("appendPrefillCacheArgs(nil, %q) = %v, want %v", cachePath, got, want)
	}
	mustExist(t, cachePath)

	// A path that cannot be created disables persistence instead of failing
	// the load.
	blocked := filepath.Join(t.TempDir(), "blocked")
	mustWriteFile(t, blocked, "")
	if got := appendPrefillCacheArgs(nil, filepath.Join(blocked, "cache")); len(got) != 0 {
		t.Fatalf("appendPrefillCacheArgs(nil, <uncreatable>) = %v, want no args", got)
	}
}
