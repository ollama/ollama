package gguf

import (
	"fmt"
	"path"
	"regexp"
	"strconv"
)

// splitShardNameRe matches the llama.cpp split GGUF shard filename pattern,
// e.g. "model-00001-of-00002.gguf".
var splitShardNameRe = regexp.MustCompile(`^(.*)-(\d{5})-of-(\d{5})\.gguf$`)

// SplitShardName parses a llama.cpp split GGUF shard filename such as
// "model-00001-of-00002.gguf". It returns the shard prefix, the zero-based
// shard index, the total shard count, and whether the name matched the split
// pattern. The name may include directories; only the base name is parsed.
func SplitShardName(name string) (prefix string, index, count int, ok bool) {
	matches := splitShardNameRe.FindStringSubmatch(path.Base(name))
	if len(matches) != 4 {
		return "", 0, 0, false
	}

	shard, err := strconv.Atoi(matches[2])
	if err != nil || shard <= 0 {
		return "", 0, 0, false
	}
	total, err := strconv.Atoi(matches[3])
	if err != nil || total <= 0 {
		return "", 0, 0, false
	}
	return matches[1], shard - 1, total, true
}

// SplitShardFileName returns the llama.cpp split GGUF shard filename for the
// given prefix, zero-based shard index, and total shard count, e.g.
// SplitShardFileName("model", 0, 2) == "model-00001-of-00002.gguf".
func SplitShardFileName(prefix string, index, count int) string {
	return fmt.Sprintf("%s-%05d-of-%05d.gguf", prefix, index+1, count)
}
