package create

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"

	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

const deepSeekV41SupportedCompressedVocabSize = 99092

var deepSeekV41Whitespace = regexp.MustCompile(`[ \t\r\n]+`)

// The publisher's engram hash uses NumPy-seeded multipliers derived from the
// compressed vocabulary size and layer IDs. These values support the layout
// below; other layouts must not silently reuse them.
var deepSeekV41Multipliers = [][]int64{
	{76632096046245, 4839876093313, 35959672319349, 73987337458391},
	{67716810739261, 51510806800915, 30921347202721, 82619226485591},
}

type deepSeekV41EngramMetadata struct {
	TokenizerSHA256     string    `json:"tokenizer_sha256"`
	TokenMapSHA256      string    `json:"token_map_sha256"`
	TokenMapEncoding    string    `json:"token_map_encoding"`
	TokenMap            []byte    `json:"token_map"`
	CompressedVocabSize int       `json:"compressed_vocab_size"`
	LayerIDs            []int     `json:"layer_ids"`
	Multipliers         [][]int64 `json:"multipliers"`
}

func importDeepSeekV41EngramMetadata(ctx context.Context, modelDir string, store BlobStore) (LayerInfo, error) {
	if err := checkContext(ctx); err != nil {
		return LayerInfo{}, err
	}
	tokenizerJSON, err := os.ReadFile(filepath.Join(modelDir, "tokenizer.json"))
	if err != nil {
		return LayerInfo{}, fmt.Errorf("read DeepSeek V4.1 tokenizer: %w", err)
	}
	configJSON, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return LayerInfo{}, fmt.Errorf("read DeepSeek V4.1 config: %w", err)
	}
	metadata, err := makeDeepSeekV41EngramMetadata(tokenizerJSON, configJSON)
	if err != nil {
		return LayerInfo{}, err
	}
	if err := checkContext(ctx); err != nil {
		return LayerInfo{}, err
	}
	return store.WriteBlob(bytes.NewReader(metadata), mediaTypeImageJSON, "engram_metadata.json")
}

// makeDeepSeekV41EngramMetadata derives the compressed vocabulary from the
// supplied tokenizer and records its digests in the model artifact.
func makeDeepSeekV41EngramMetadata(tokenizerJSON, configJSON []byte) ([]byte, error) {
	var config struct {
		Text struct {
			VocabSize           int   `json:"vocab_size"`
			CompressedVocabSize int   `json:"engram_compressed_vocab_size"`
			LayerIDs            []int `json:"engram_layer_ids"`
			MaxNGramSize        int   `json:"engram_max_ngram_size"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(configJSON, &config); err != nil {
		return nil, fmt.Errorf("parse DeepSeek V4.1 config: %w", err)
	}
	text := config.Text
	if text.VocabSize <= 0 || text.CompressedVocabSize != deepSeekV41SupportedCompressedVocabSize ||
		!slices.Equal(text.LayerIDs, []int{1, 14}) || text.MaxNGramSize != len(deepSeekV41Multipliers[0]) {
		return nil, fmt.Errorf("unsupported DeepSeek V4.1 engram hash layout: compressed vocabulary %d, layers %v, n-gram size %d", text.CompressedVocabSize, text.LayerIDs, text.MaxNGramSize)
	}
	tok, err := tokenizer.LoadFromBytes(tokenizerJSON)
	if err != nil {
		return nil, fmt.Errorf("load DeepSeek V4.1 tokenizer: %w", err)
	}
	if tok.VocabSize() != text.VocabSize {
		return nil, fmt.Errorf("DeepSeek V4.1 tokenizer has %d IDs, want %d", tok.VocabSize(), text.VocabSize)
	}
	var source struct {
		Model struct {
			Vocab map[string]int32 `json:"vocab"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int32  `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(tokenizerJSON, &source); err != nil {
		return nil, fmt.Errorf("parse DeepSeek V4.1 tokenizer vocabulary: %w", err)
	}
	rawByID := make([]string, text.VocabSize)
	added := make(map[int32]string, len(source.AddedTokens))
	for value, id := range source.Model.Vocab {
		if id < 0 || int(id) >= len(rawByID) {
			return nil, fmt.Errorf("DeepSeek V4.1 vocabulary ID %d is out of range", id)
		}
		rawByID[id] = value
	}
	for _, token := range source.AddedTokens {
		if token.ID < 0 || int(token.ID) >= len(rawByID) {
			return nil, fmt.Errorf("DeepSeek V4.1 added token ID %d is out of range", token.ID)
		}
		added[token.ID] = token.Content
		rawByID[token.ID] = token.Content
	}
	mapBytes := make([]byte, text.VocabSize*4)
	assigned := make(map[string]int32, text.VocabSize)
	for id, raw := range rawByID {
		if raw == "" {
			return nil, fmt.Errorf("DeepSeek V4.1 tokenizer lacks token ID %d", id)
		}
		decoded, isAdded := added[int32(id)]
		if !isAdded {
			decoded = strings.ToValidUTF8(tok.Decode([]int32{int32(id)}), "\ufffd")
		}
		key := raw
		if !strings.ContainsRune(decoded, '\ufffd') {
			key = normalizeDeepSeekV41Token(decoded)
			if key == "" {
				key = decoded
			}
		}
		compressed, ok := assigned[key]
		if !ok {
			compressed = int32(len(assigned))
			assigned[key] = compressed
		}
		binary.LittleEndian.PutUint32(mapBytes[id*4:], uint32(compressed))
	}
	if len(assigned) != text.CompressedVocabSize {
		return nil, fmt.Errorf("DeepSeek V4.1 tokenizer produces %d compressed IDs, config expects %d", len(assigned), text.CompressedVocabSize)
	}
	metadata := deepSeekV41EngramMetadata{
		TokenizerSHA256:     fmt.Sprintf("%x", sha256.Sum256(tokenizerJSON)),
		TokenMapSHA256:      fmt.Sprintf("%x", sha256.Sum256(mapBytes)),
		TokenMapEncoding:    "little-endian-int32-base64",
		TokenMap:            mapBytes,
		CompressedVocabSize: len(assigned),
		LayerIDs:            text.LayerIDs,
		Multipliers:         deepSeekV41Multipliers,
	}
	data, err := json.Marshal(metadata)
	if err != nil {
		return nil, fmt.Errorf("marshal DeepSeek V4.1 engram metadata: %w", err)
	}
	return append(data, '\n'), nil
}

func normalizeDeepSeekV41Token(value string) string {
	value = norm.NFD.String(norm.NFKC.String(value))
	value = strings.Map(func(r rune) rune {
		if unicode.Is(unicode.M, r) {
			return -1
		}
		return r
	}, value)
	value = deepSeekV41Whitespace.ReplaceAllString(strings.ToLower(value), " ")
	if value != " " {
		value = strings.TrimSpace(value)
	}
	return value
}
