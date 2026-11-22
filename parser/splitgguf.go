package parser

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/gguf"
)

// expandSplitGGUFShards expands references to llama.cpp split GGUF shards
// (e.g. "model-00001-of-00002.gguf") so that every shard of the model is
// gathered, not just the referenced one. Referencing any single shard of a
// split GGUF imports the whole model, mirroring `llama-gguf-split --merge`.
//
// Expansion only applies to files that exist and whose GGUF metadata declares
// more than one shard (split.count > 1); anything else is left untouched so
// remote model references and single-file GGUFs keep working. If a shard of a
// split GGUF is absent from the shard's directory, an error naming the missing
// shard(s) is returned.
func expandSplitGGUFShards(paths []string) ([]string, error) {
	seen := make(map[string]struct{}, len(paths))
	expanded := make([]string, 0, len(paths))

	add := func(p string) {
		if _, ok := seen[p]; ok {
			return
		}
		seen[p] = struct{}{}
		expanded = append(expanded, p)
	}

	for _, p := range paths {
		add(p)

		prefix, _, count, ok := gguf.SplitShardName(p)
		if !ok || count <= 1 {
			continue
		}
		if !isSplitGGUFFile(p) {
			// Not a split GGUF: a missing path (handled later as a remote
			// model reference) or a single-file GGUF whose name merely
			// matches the shard pattern.
			continue
		}

		dir := filepath.Dir(p)
		var missing []string
		for i := range count {
			shard := filepath.Join(dir, gguf.SplitShardFileName(prefix, i, count))
			if _, err := os.Stat(shard); errors.Is(err, os.ErrNotExist) {
				missing = append(missing, filepath.Base(shard))
				continue
			} else if err != nil {
				return nil, err
			}
			add(shard)
		}
		if len(missing) > 0 {
			slices.Sort(missing)
			return nil, fmt.Errorf("split GGUF %s is missing shard(s) in %s: %s", filepath.Base(p), dir, strings.Join(missing, ", "))
		}
	}

	slices.Sort(expanded)
	return expanded, nil
}

// isSplitGGUFFile reports whether path is a GGUF file whose metadata declares
// more than one shard (split.count > 1).
func isSplitGGUFFile(path string) bool {
	metadata, err := gguf.ReadFileMetadata(path, 1)
	if err != nil {
		return false
	}
	count, ok := metadata.UintOK("split.count")
	return ok && count > 1
}
