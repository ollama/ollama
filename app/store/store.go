package store

import (
	"encoding/json"
	"errors"
	"log"
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
			log.Printf("XXX loaded existing store - ID: %s", store.ID)
			return
		}
	}
	log.Printf("XXX initializing new store %s", err)
	store.ID = uuid.New().String()
	writeStore(getStorePath())
}

func writeStore(storeFilename string) {
	ollamaDir := filepath.Dir(storeFilename)
	_, err := os.Stat(ollamaDir)
	if errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(ollamaDir, 0o755); err != nil {
			log.Printf("create ollama dir %s: %v", ollamaDir, err)
			return
		}
		log.Printf("XXX created ollamaDir: %s", ollamaDir)
	}
	payload, err := json.Marshal(store)
	fp, err := os.OpenFile(storeFilename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
	if err != nil {
		log.Printf("write store payload %s: %v", storeFilename, err)
		return
	}
	defer fp.Close()
	if n, err := fp.Write(payload); err != nil || n != len(payload) {
		log.Printf("write store payload %s: %d vs %d -- %v", storeFilename, n, len(payload), err)
		return
	}
	log.Printf("XXX wrote store: %s", storeFilename)
}
