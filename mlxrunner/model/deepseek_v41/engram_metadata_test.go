package deepseek_v41

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"testing"
)

func engramMetadataFixture(t interface {
	Helper()
	Fatal(...any)
},
) ([]byte, []byte, Config) {
	t.Helper()
	cfg := Config{VocabSize: 3, EngramCompressedVocabSize: 2, EngramLayerIDs: []int{1}, EngramMaxNGramSize: 2}
	tokenizer := []byte(`{"test":"tokenizer"}`)
	rawMap := make([]byte, 12)
	for i, id := range []uint32{0, 1, 0} {
		binary.LittleEndian.PutUint32(rawMap[i*4:], id)
	}
	metadata := engramMetadata{
		TokenizerSHA256: fmt.Sprintf("%x", sha256.Sum256(tokenizer)), TokenMapSHA256: fmt.Sprintf("%x", sha256.Sum256(rawMap)),
		TokenMapEncoding: "little-endian-int32-base64", TokenMap: rawMap, CompressedVocabSize: 2,
		LayerIDs: []int{1}, Multipliers: [][]int64{{3, 7}},
	}
	data, err := json.Marshal(metadata)
	if err != nil {
		t.Fatal(err)
	}
	return data, tokenizer, cfg
}

func TestEngramMetadataBindsTokenizerAndMap(t *testing.T) {
	data, tokenizer, cfg := engramMetadataFixture(t)
	ids, multipliers, err := parseEngramMetadata(data, tokenizer, &cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 3 || ids[1] != 1 || len(multipliers) != 1 || multipliers[0][1] != 7 {
		t.Fatalf("unexpected engram metadata: %v, %v", ids, multipliers)
	}
	if _, _, err := parseEngramMetadata(data, []byte(`{"test":"other"}`), &cfg); err == nil {
		t.Fatal("accepted metadata for a different tokenizer")
	}
}

func FuzzParseEngramMetadata(f *testing.F) {
	seed, tokenizer, cfg := engramMetadataFixture(f)
	f.Add(seed)
	f.Add([]byte(`{"token_map":"AA=="}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1<<20 {
			t.Skip("metadata exceeds fuzz input budget")
		}
		ids, multipliers, err := parseEngramMetadata(data, tokenizer, &cfg)
		if err != nil {
			return
		}
		if len(ids) != cfg.VocabSize || len(multipliers) != len(cfg.EngramLayerIDs) {
			t.Fatalf("accepted metadata with dimensions %d IDs and %d multiplier sets", len(ids), len(multipliers))
		}
		for _, id := range ids {
			if id < 0 || int(id) >= cfg.EngramCompressedVocabSize {
				t.Fatalf("accepted compressed token ID %d outside range", id)
			}
		}
	})
}
