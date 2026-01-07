package server

import (
	"encoding/json"
	"errors"
	"os"
)

// ShardMetadata stores information about sharded GGUF models
// Stored alongside blob: sha256-<digest>.shardmeta.json
type ShardMetadata struct {
	IsSharded    bool     `json:"is_sharded"`
	ShardIndex   int      `json:"shard_index"` // 1-indexed
	ShardCount   int      `json:"shard_count"`
	ShardDigests []string `json:"shard_digests"` // All shard digests in order
	BaseName     string   `json:"base_name"`     // Original basename (e.g., "model")
}

// WriteShardMetadata stores shard info for a model layer
func WriteShardMetadata(digest string, meta ShardMetadata) error {
	blobPath, err := GetBlobsPath(digest)
	if err != nil {
		return err
	}

	metaPath := blobPath + ".shardmeta.json"

	f, err := os.Create(metaPath)
	if err != nil {
		return err
	}
	defer f.Close()

	return json.NewEncoder(f).Encode(meta)
}

// ReadShardMetadata loads shard info if it exists
func ReadShardMetadata(digest string) (*ShardMetadata, error) {
	blobPath, err := GetBlobsPath(digest)
	if err != nil {
		return nil, err
	}

	metaPath := blobPath + ".shardmeta.json"

	f, err := os.Open(metaPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil // No metadata = not sharded
		}
		return nil, err
	}
	defer f.Close()

	var meta ShardMetadata
	if err := json.NewDecoder(f).Decode(&meta); err != nil {
		return nil, err
	}

	return &meta, nil
}

// DeleteShardMetadata removes shard metadata file
func DeleteShardMetadata(digest string) error {
	blobPath, err := GetBlobsPath(digest)
	if err != nil {
		return err
	}

	metaPath := blobPath + ".shardmeta.json"
	err = os.Remove(metaPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil // Already deleted
	}
	return err
}
