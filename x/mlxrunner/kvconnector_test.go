package mlxrunner

import (
	"fmt"
	"slices"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/cache"
)

type memoryKVConnector struct {
	entries map[string]*KVConnectorMatch
	stores  int
}

func newMemoryKVConnector() *memoryKVConnector {
	return &memoryKVConnector{entries: make(map[string]*KVConnectorMatch)}
}

func (c *memoryKVConnector) Lookup(keys []trieKey) (*KVConnectorMatch, error) {
	for offset := len(keys); offset > 0; offset-- {
		if match := c.entries[memoryConnectorKey(keys[:offset])]; match != nil {
			return &KVConnectorMatch{
				Offset:    match.Offset,
				Snapshots: cloneFakeSnapshots(match.Snapshots),
			}, nil
		}
	}
	return nil, nil
}

func (c *memoryKVConnector) SnapshotOffsets(inputs []int32, draftLookahead int) []int {
	if len(inputs) <= 1 {
		return nil
	}
	return []int{len(inputs) - 1 + draftLookahead}
}

func (c *memoryKVConnector) Store(entry *KVConnectorEntry) error {
	c.stores++
	c.entries[memoryConnectorKey(entry.Keys)] = &KVConnectorMatch{
		Offset:    entry.Offset,
		Snapshots: cloneFakeSnapshots(entry.Snapshots),
	}
	return nil
}

func memoryConnectorKey(keys []trieKey) string {
	return fmt.Sprint(keys)
}

func cloneFakeSnapshots(snaps []cache.Snapshot) []cache.Snapshot {
	out := make([]cache.Snapshot, len(snaps))
	for i, snap := range snaps {
		if snap == nil {
			continue
		}
		fs := snap.(*fakeSnapshot)
		out[i] = &fakeSnapshot{
			tokens:   slices.Clone(fs.tokens),
			from:     fs.from,
			to:       fs.to,
			byteSize: fs.byteSize,
		}
	}
	return out
}

func TestKVConnectorPersistsAndRestoresPromptPrefix(t *testing.T) {
	inputs := []int32{1, 2, 3, 4, 5}
	connector := newMemoryKVConnector()

	storeEnv := newTransformerEnv()
	storeEnv.pc.connector = connector

	storeSession := storeEnv.pc.begin(inputs, nil)
	storeSession.schedulePrefillSnapshots(nil)
	seed := len(inputs) - 1
	if base := storeEnv.pc.minCacheOffset(); base < seed {
		feedAll(storeEnv.pc.caches, inputs[base:seed])
	}
	storeSession.attachPrefillSnapshots()
	storeSession.close()

	if connector.stores == 0 {
		t.Fatal("connector did not store any snapshots")
	}

	restoreEnv := newTransformerEnv()
	restoreEnv.pc.connector = connector

	restoreSession := restoreEnv.pc.begin(inputs, nil)
	defer restoreSession.close()

	if want := []int32{5}; !slices.Equal(restoreSession.remaining, want) {
		t.Fatalf("remaining = %v, want %v", restoreSession.remaining, want)
	}

	restoreEnv.assertAllTokens(t, "restored prefix", inputs[:4])
}
