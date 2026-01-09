package parser

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strconv"
)

// ShardPattern matches: model-00001-of-00003.gguf
var shardPattern = regexp.MustCompile(`^(.+)-(\d{5})-of-(\d{5})\.gguf$`)

type GGUFShardSet struct {
	BaseName   string   // "model" or full path prefix
	TotalCount int      // Total number of shards
	Shards     []string // Ordered list of shard file paths
}

// GroupGGUFShards separates sharded GGUF files from single files
func GroupGGUFShards(files []string) ([]GGUFShardSet, []string) {
	type shardInfo struct {
		totalCount int
		shards     map[int]string // index -> filepath
	}

	shardMap := make(map[string]*shardInfo) // basename -> shardInfo
	var singles []string

	for _, file := range files {
		base := filepath.Base(file)
		matches := shardPattern.FindStringSubmatch(base)

		if matches == nil {
			// Not a shard, single GGUF file
			singles = append(singles, file)
			continue
		}

		basename := matches[1]
		index, _ := strconv.Atoi(matches[2])
		totalCount, _ := strconv.Atoi(matches[3])

		if shardMap[basename] == nil {
			shardMap[basename] = &shardInfo{
				totalCount: totalCount,
				shards:     make(map[int]string),
			}
		}

		// Verify total count is consistent
		if shardMap[basename].totalCount != totalCount {
			// Inconsistent total count - treat as separate single files
			singles = append(singles, file)
			continue
		}

		shardMap[basename].shards[index] = file
	}

	// Convert map to slice of ShardSets
	var shardSets []GGUFShardSet
	for basename, info := range shardMap {
		shards := make([]string, 0, len(info.shards))

		// Sort by index and build ordered slice
		for i := 1; i <= info.totalCount; i++ {
			if path, ok := info.shards[i]; ok {
				shards = append(shards, path)
			}
		}

		shardSets = append(shardSets, GGUFShardSet{
			BaseName:   basename,
			TotalCount: info.totalCount,
			Shards:     shards,
		})
	}

	return shardSets, singles
}

// validateShardSet ensures all shards are present and correctly named
func validateShardSet(set GGUFShardSet) error {
	if len(set.Shards) != set.TotalCount {
		return fmt.Errorf("expected %d shards, found %d", set.TotalCount, len(set.Shards))
	}

	// Verify sequential numbering by checking filenames
	for i, shard := range set.Shards {
		expected := fmt.Sprintf("%s-%05d-of-%05d.gguf", set.BaseName, i+1, set.TotalCount)
		actual := filepath.Base(shard)
		if actual != expected {
			return fmt.Errorf("shard %d has unexpected name: %s (expected: %s)", i+1, actual, expected)
		}
	}

	return nil
}

// findShardSet checks if a file is part of a shard set in the file list
func findShardSet(file string, allFiles []string) *GGUFShardSet {
	base := filepath.Base(file)
	matches := shardPattern.FindStringSubmatch(base)

	if matches == nil {
		return nil // Not a shard file
	}

	basename := matches[1]

	// Find all shards with the same basename
	var shards []string
	for _, f := range allFiles {
		fbase := filepath.Base(f)
		if fmatches := shardPattern.FindStringSubmatch(fbase); fmatches != nil {
			if fmatches[1] == basename {
				shards = append(shards, f)
			}
		}
	}

	if len(shards) == 0 {
		return nil
	}

	// Use GroupGGUFShards to properly sort and validate
	shardSets, _ := GroupGGUFShards(shards)
	if len(shardSets) > 0 {
		return &shardSets[0]
	}

	return nil
}
