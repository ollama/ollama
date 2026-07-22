package server

import (
	"fmt"
	"os"
	"sync"

	"github.com/ollama/ollama/fs/gguf"
)

// ggufMetadata holds the GGUF values the server reads during capability
// detection and template selection. They are properties of the blob itself,
// identical regardless of which model references it, so they can be cached and
// shared.
type ggufMetadata struct {
	architecture string
	chatTemplate string
	hasPooling   bool
	hasVision    bool
	hasAudio     bool
}

// ggufMetadataCache maps a blob identity to its derived metadata so a given
// GGUF file is read at most once. Blobs are content-addressed, so a re-pull with
// changed metadata yields a new key; size and mtime guard against a file being
// replaced in place. Like modelShowCache, it is process-local, bounded in
// practice by the number of distinct blobs loaded, and cleared on restart.
var ggufMetadataCache sync.Map // map[string]*ggufMetadata

// ggufBlobKey identifies a blob by path, size and mtime. The path alone is
// insufficient because callers such as ggufCapabilities only have the model
// path, not the layer digest.
func ggufBlobKey(path string, fi os.FileInfo) string {
	return fmt.Sprintf("%s|%d|%d", path, fi.Size(), fi.ModTime().UnixNano())
}

// loadGGUFMetadata returns the derived metadata for the GGUF blob at path,
// reading the file the first time a blob is seen and serving subsequent calls
// from the cache.
func loadGGUFMetadata(path string) (*ggufMetadata, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, err
	}

	key := ggufBlobKey(path, fi)
	if v, ok := ggufMetadataCache.Load(key); ok {
		return v.(*ggufMetadata), nil
	}

	f, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// architecture must be read first: KeyValue prefixes non-general,
	// non-tokenizer keys with it. Absent keys (e.g. pooling_type on a model with
	// no pooling) scan to the end of the metadata block, but the reader caches
	// decoded values on f, so the file is scanned at most once here.
	md := &ggufMetadata{
		architecture: f.KeyValue("general.architecture").String(),
		chatTemplate: f.KeyValue("tokenizer.chat_template").String(),
		hasPooling:   f.KeyValue("pooling_type").Valid(),
		hasVision:    f.KeyValue("vision.block_count").Valid(),
		hasAudio:     f.KeyValue("audio.block_count").Valid(),
	}

	// Concurrent first requests for the same uncached blob may each parse it,
	// but LoadOrStore keeps a single shared value. This happens at most once per
	// blob per process, so it is not worth singleflighting.
	actual, _ := ggufMetadataCache.LoadOrStore(key, md)
	return actual.(*ggufMetadata), nil
}
