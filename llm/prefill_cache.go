// prefill_cache.go persists a llama-server runner's processed prompt across
// unload and reload, by driving the slot save and restore that llama-server
// already exposes on its internal /slots endpoint. llamaServerRunner satisfies
// PrefillCachePersistor through the two exported methods here.
//
// The scheduler owns the lifecycle: it chooses the directory, hands it over as
// LlamaServerConfig.PrefillCachePath, saves before a runner shuts down, and
// restores once its replacement is running. An empty path disables persistence,
// which is the default.
//
// Key properties:
//   - Both directions hold every slot of the runner's semaphore, so they only
//     ever run against a quiesced server.
//   - manifest.json names the slots that hold a snapshot, with a checksum for
//     each, and is replaced last, which makes it the commit point of a save.
//   - Failures are returned rather than fatal so the caller can fall back to a
//     cold prefill, and only a snapshot that is provably unusable is deleted.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// llamaPrefillCacheEraseTimeout bounds the slot erase that follows a failed
// restore. The erase is best effort cleanup on a runner that is about to
// serve requests, so it is kept short: leaving the slot dirty is recorded
// as a warning, but blocking the load behind a wedged server is not.
const llamaPrefillCacheEraseTimeout = 10 * time.Second

type llamaPrefillCacheManifest struct {
	Slots []llamaPrefillCacheSlot `json:"slots"`
}

type llamaPrefillCacheSlot struct {
	ID  int    `json:"id"`
	CRC uint32 `json:"crc32c"`
}

var errInvalidLlamaPrefillCache = errors.New("invalid llama prefill cache")

// llamaPrefillCacheCRC is Castagnoli, which compiles down to a hardware
// instruction on amd64 and arm64. The checksum only has to catch a snapshot
// damaged after it was written, and the manifest holding it sits in the same
// directory under the same permissions, so a cryptographic digest would defend
// against nothing this does not while costing several times as much per byte.
var llamaPrefillCacheCRC = crc32.MakeTable(crc32.Castagnoli)

func checksumPrefillCacheSlot(path string) (uint32, int64, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, 0, err
	}
	defer file.Close()
	hash := crc32.New(llamaPrefillCacheCRC)
	size, err := io.Copy(hash, file)
	if err != nil {
		return 0, 0, err
	}
	return hash.Sum32(), size, nil
}

func (s *llamaServerRunner) SavePrefillCache(ctx context.Context) error {
	cachePath := s.launch.config.PrefillCachePath
	if cachePath == "" {
		return nil
	}
	// Check before creating anything: a canceled save must not rebuild a
	// directory the daemon's exit cleanup is removing.
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := os.MkdirAll(cachePath, 0o700); err != nil {
		return fmt.Errorf("create prefill cache directory: %w", err)
	}

	if err := s.sem.Acquire(ctx, int64(s.launch.numParallel)); err != nil {
		return err
	}
	defer s.sem.Release(int64(s.launch.numParallel))

	start := time.Now()
	var savedBytes int64
	savedSlots := make([]llamaPrefillCacheSlot, 0, s.launch.numParallel)
	for id := range s.launch.numParallel {
		finalName := fmt.Sprintf("slot-%d.bin", id)
		tempName := finalName + ".tmp"
		finalPath := filepath.Join(cachePath, finalName)
		tempPath := filepath.Join(cachePath, tempName)
		result, err := s.slotCacheAction(ctx, id, "save", tempName)
		if err != nil {
			var httpErr *llamaSlotCacheHTTPError
			if errors.As(err, &httpErr) && httpErr.StatusCode == http.StatusNotImplemented {
				_ = os.Remove(tempPath)
				if err := os.Remove(finalPath); err != nil && !os.IsNotExist(err) {
					return fmt.Errorf("remove unsupported slot %d prefill cache: %w", id, err)
				}
				continue
			}
			return err
		}
		if result.NSaved == 0 {
			_ = os.Remove(tempPath)
			if err := os.Remove(finalPath); err != nil && !os.IsNotExist(err) {
				return fmt.Errorf("remove empty slot %d prefill cache: %w", id, err)
			}
			continue
		}
		if err := os.Remove(finalPath); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("replace slot %d prefill cache: %w", id, err)
		}
		if err := os.Rename(tempPath, finalPath); err != nil {
			return fmt.Errorf("commit slot %d prefill cache: %w", id, err)
		}
		crc, size, err := checksumPrefillCacheSlot(finalPath)
		if err != nil {
			return fmt.Errorf("checksum slot %d prefill cache: %w", id, err)
		}
		savedBytes += size
		savedSlots = append(savedSlots, llamaPrefillCacheSlot{ID: id, CRC: crc})
	}

	manifest, err := json.Marshal(llamaPrefillCacheManifest{Slots: savedSlots})
	if err != nil {
		return err
	}
	// Replace the manifest last, so a save that fails partway leaves the
	// previous generation listed and restorable. Slot files are replaced one
	// at a time and each is self-consistent, so a mixed generation restores a
	// shorter prefix rather than a wrong one.
	tempManifest := filepath.Join(cachePath, "manifest.json.tmp")
	if err := os.WriteFile(tempManifest, manifest, 0o600); err != nil {
		return fmt.Errorf("write prefill cache manifest: %w", err)
	}
	if err := os.Rename(tempManifest, filepath.Join(cachePath, "manifest.json")); err != nil {
		return fmt.Errorf("commit prefill cache manifest: %w", err)
	}
	if len(savedSlots) > 0 {
		slog.Info("saved prefill cache", "slots", len(savedSlots), "bytes", savedBytes, "duration", time.Since(start))
	}
	return nil
}

func (s *llamaServerRunner) RestorePrefillCache(ctx context.Context) error {
	err := s.restorePrefillCache(ctx)
	if errors.Is(err, errInvalidLlamaPrefillCache) && s.launch.config.PrefillCachePath != "" {
		_ = os.RemoveAll(s.launch.config.PrefillCachePath)
	}
	return err
}

func (s *llamaServerRunner) restorePrefillCache(ctx context.Context) error {
	cachePath := s.launch.config.PrefillCachePath
	if cachePath == "" {
		return nil
	}

	data, err := os.ReadFile(filepath.Join(cachePath, "manifest.json"))
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read prefill cache manifest: %w", err)
	}
	var manifest llamaPrefillCacheManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return fmt.Errorf("%w: decode manifest: %v", errInvalidLlamaPrefillCache, err)
	}
	seen := make(map[int]bool, len(manifest.Slots))
	for _, slot := range manifest.Slots {
		if slot.ID < 0 || slot.ID >= s.launch.numParallel || seen[slot.ID] {
			return fmt.Errorf("%w: incompatible slot %d", errInvalidLlamaPrefillCache, slot.ID)
		}
		seen[slot.ID] = true
	}
	// Verify every slot before the server reads any of them. llama-server
	// checks a save file's header against the model it is loaded with but not
	// the payload behind it, so a snapshot damaged after it was written
	// restores as garbage and the reply is wrong rather than merely cold.
	// This also makes a half-saved generation a cold miss.
	verifyStart := time.Now()
	for _, slot := range manifest.Slots {
		crc, _, err := checksumPrefillCacheSlot(filepath.Join(cachePath, fmt.Sprintf("slot-%d.bin", slot.ID)))
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("%w: slot %d is missing", errInvalidLlamaPrefillCache, slot.ID)
		}
		if err != nil {
			return fmt.Errorf("read slot %d prefill cache: %w", slot.ID, err)
		}
		if crc != slot.CRC {
			return fmt.Errorf("%w: slot %d checksum mismatch", errInvalidLlamaPrefillCache, slot.ID)
		}
	}
	verify := time.Since(verifyStart)

	if err := s.sem.Acquire(ctx, int64(s.launch.numParallel)); err != nil {
		return err
	}
	defer s.sem.Release(int64(s.launch.numParallel))

	start := time.Now()
	for _, slot := range manifest.Slots {
		if _, err := s.slotCacheAction(ctx, slot.ID, "restore", fmt.Sprintf("slot-%d.bin", slot.ID)); err != nil {
			// Clear any partial state before falling back to cold prefill.
			// The snapshot is left alone: llama-server answers 400 for a bad
			// save file and for runtime failures such as insufficient KV
			// space, so its status cannot prove the file is corrupt.
			s.eraseSlotAfterFailedRestore(slot.ID)
			return err
		}
	}
	if len(manifest.Slots) > 0 {
		slog.Info("restored prefill cache", "slots", len(manifest.Slots), "verify", verify, "duration", time.Since(start))
	}
	return nil
}

// eraseSlotAfterFailedRestore uses its own context: the restore may have
// failed precisely because the caller's context was canceled, and the erase
// must still run to clear any partially restored state.
func (s *llamaServerRunner) eraseSlotAfterFailedRestore(id int) {
	ctx, cancel := context.WithTimeout(context.Background(), llamaPrefillCacheEraseTimeout)
	defer cancel()
	if _, err := s.slotCacheAction(ctx, id, "erase", ""); err != nil {
		slog.Warn("failed to erase slot after failed prefill cache restore; slot may hold partial state", "slot", id, "error", err)
	}
}

type llamaSlotCacheHTTPError struct {
	StatusCode int
	Action     string
	Slot       int
	Message    string
}

func (e *llamaSlotCacheHTTPError) Error() string {
	return fmt.Sprintf("%s slot %d prefill cache: status %d: %s", e.Action, e.Slot, e.StatusCode, e.Message)
}

type llamaSlotCacheActionResponse struct {
	NSaved int `json:"n_saved"`
}

func (s *llamaServerRunner) slotCacheAction(ctx context.Context, id int, action, filename string) (llamaSlotCacheActionResponse, error) {
	body, err := json.Marshal(map[string]string{"filename": filename})
	if err != nil {
		return llamaSlotCacheActionResponse{}, err
	}
	url := fmt.Sprintf("http://127.0.0.1:%d/slots/%d?action=%s", s.port, id, action)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return llamaSlotCacheActionResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.httpClient().Do(req)
	if err != nil {
		return llamaSlotCacheActionResponse{}, fmt.Errorf("%s slot %d prefill cache: %w", action, id, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		message, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		return llamaSlotCacheActionResponse{}, &llamaSlotCacheHTTPError{
			StatusCode: resp.StatusCode,
			Action:     action,
			Slot:       id,
			Message:    string(bytes.TrimSpace(message)),
		}
	}
	var result llamaSlotCacheActionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return llamaSlotCacheActionResponse{}, fmt.Errorf("decode %s slot %d prefill cache response: %w", action, id, err)
	}
	return result, nil
}
