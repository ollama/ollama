package store

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"

	"github.com/google/uuid"
)

type Store struct {
	ID           string `json:"id"`
	FirstTimeRun bool   `json:"first-time-run"`
}

var (
	lock  sync.Mutex
	store Store
)

func GetID() string {
	lock.Lock()
	defer lock.Unlock()
	if store.ID == "" {
		initStore()
	}
	return store.ID

}

func GetFirstTimeRun() bool {
	lock.Lock()
	defer lock.Unlock()
	if store.ID == "" {
		initStore()
	}
	return store.FirstTimeRun
}

func SetFirstTimeRun(val bool) {
	lock.Lock()
	defer lock.Unlock()
	if store.FirstTimeRun == val {
		return
	}
	store.FirstTimeRun = val
	writeStore(getStorePath())
}

// lock must be held
func initStore() {
	storeFile, err := os.Open(getStorePath())
	if err == nil {
		defer storeFile.Close()
		err = json.NewDecoder(storeFile).Decode(&store)
		if err == nil {
			slog.Debug(fmt.Sprintf("loaded existing store %s - ID: %s", getStorePath(), store.ID))
			return
		}
	}
	slog.Debug(fmt.Sprintf("XXX initializing new store %s", err))
	store.ID = uuid.New().String()
	writeStore(getStorePath())
}

func writeStore(storeFilename string) {
	ollamaDir := filepath.Dir(storeFilename)
	_, err := os.Stat(ollamaDir)
	if errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(ollamaDir, 0o755); err != nil {
			slog.Debug(fmt.Sprintf("create ollama dir %s: %v", ollamaDir, err))
			return
		}
		slog.Debug(fmt.Sprintf("XXX created ollamaDir: %s", ollamaDir))
	}
	payload, err := json.Marshal(store)
	if err != nil {
		slog.Error(fmt.Sprintf("failed to marshal store: %s", err))
		return
	}
	fp, err := os.OpenFile(storeFilename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
	if err != nil {
		slog.Error(fmt.Sprintf("write store payload %s: %v", storeFilename, err))
		return
	}
	defer fp.Close()
	if n, err := fp.Write(payload); err != nil || n != len(payload) {
		slog.Error(fmt.Sprintf("write store payload %s: %d vs %d -- %v", storeFilename, n, len(payload), err))
		return
	}
	slog.Debug("Store: " + string(payload))
	slog.Info(fmt.Sprintf("wrote store: %s", storeFilename))
}
