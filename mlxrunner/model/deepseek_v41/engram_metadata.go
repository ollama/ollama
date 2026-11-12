package deepseek_v41

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"slices"
)

type engramMetadata struct {
	TokenizerSHA256     string    `json:"tokenizer_sha256"`
	TokenMapSHA256      string    `json:"token_map_sha256"`
	TokenMapEncoding    string    `json:"token_map_encoding"`
	TokenMap            []byte    `json:"token_map"`
	CompressedVocabSize int       `json:"compressed_vocab_size"`
	LayerIDs            []int     `json:"layer_ids"`
	Multipliers         [][]int64 `json:"multipliers"`
}

func parseEngramMetadata(data, tokenizerJSON []byte, cfg *Config) ([]int32, [][]int64, error) {
	var metadata engramMetadata
	if err := json.Unmarshal(data, &metadata); err != nil {
		return nil, nil, fmt.Errorf("parse engram metadata: %w", err)
	}
	if metadata.TokenMapEncoding != "little-endian-int32-base64" ||
		metadata.CompressedVocabSize != cfg.EngramCompressedVocabSize ||
		!slices.Equal(metadata.LayerIDs, cfg.EngramLayerIDs) ||
		len(metadata.TokenMap) != cfg.VocabSize*4 ||
		len(metadata.Multipliers) != len(cfg.EngramLayerIDs) {
		return nil, nil, fmt.Errorf("engram metadata does not match the DeepSeek V4.1 model config")
	}
	if fmt.Sprintf("%x", sha256.Sum256(tokenizerJSON)) != metadata.TokenizerSHA256 ||
		fmt.Sprintf("%x", sha256.Sum256(metadata.TokenMap)) != metadata.TokenMapSHA256 {
		return nil, nil, fmt.Errorf("engram tokenizer or compressed map digest mismatch")
	}
	mapIDs := make([]int32, cfg.VocabSize)
	for i := range mapIDs {
		mapIDs[i] = int32(binary.LittleEndian.Uint32(metadata.TokenMap[4*i:]))
		if mapIDs[i] < 0 || int(mapIDs[i]) >= metadata.CompressedVocabSize {
			return nil, nil, fmt.Errorf("compressed token ID %d is out of range", mapIDs[i])
		}
	}
	for _, values := range metadata.Multipliers {
		if len(values) != cfg.EngramMaxNGramSize {
			return nil, nil, fmt.Errorf("engram multiplier count does not match model")
		}
	}
	return mapIDs, metadata.Multipliers, nil
}
