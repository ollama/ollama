package tokenizer

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"
	"unicode/utf8"
)

// Unigram encodes pre-split words with the DeBERTa SentencePiece tokenizer.
// It keeps the checkpoint's precompiled Unicode normalization table rather
// than approximating it with a different Unicode normalization form.
type Unigram struct {
	pieces       map[string]int32
	scores       []float64
	added        map[string]int32
	maxPiece     int
	unknown      int32
	unknownScore float64
	chars        []uint32
	replacements string
}

func LoadUnigram(data []byte) (*Unigram, error) {
	var raw struct {
		Model struct {
			Type         string            `json:"type"`
			Vocab        []json.RawMessage `json:"vocab"`
			Unknown      int32             `json:"unk_id"`
			ByteFallback bool              `json:"byte_fallback"`
		} `json:"model"`
		Added []struct {
			ID      int32  `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
		Normalizer struct {
			Type        string            `json:"type"`
			Normalizers []json.RawMessage `json:"normalizers"`
		} `json:"normalizer"`
		PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	if raw.Model.Type != "Unigram" || raw.Model.ByteFallback {
		return nil, fmt.Errorf("expected Unigram tokenizer without byte fallback")
	}
	if len(raw.Model.Vocab) == 0 || raw.Model.Unknown < 0 || int(raw.Model.Unknown) >= len(raw.Model.Vocab) {
		return nil, fmt.Errorf("invalid Unigram vocabulary")
	}
	u := &Unigram{pieces: make(map[string]int32), added: make(map[string]int32), unknown: raw.Model.Unknown}
	for i, entry := range raw.Model.Vocab {
		var pair []json.RawMessage
		if err := json.Unmarshal(entry, &pair); err != nil || len(pair) != 2 {
			return nil, fmt.Errorf("invalid Unigram piece %d", i)
		}
		var piece string
		var score float64
		if err := json.Unmarshal(pair[0], &piece); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(pair[1], &score); err != nil {
			return nil, err
		}
		u.pieces[piece] = int32(i)
		u.scores = append(u.scores, score)
		u.maxPiece = max(u.maxPiece, len(piece))
		u.unknownScore = min(u.unknownScore, score)
	}
	u.unknownScore -= 10
	for _, a := range raw.Added {
		u.added[a.Content] = a.ID
	}
	// Restrict the accepted pipeline to the one implemented here. Other
	// tokenizer families must never silently use DeBERTa normalization.
	if raw.Normalizer.Type != "Sequence" || len(raw.Normalizer.Normalizers) != 3 {
		return nil, fmt.Errorf("unsupported Unigram normalizer")
	}
	for i, n := range raw.Normalizer.Normalizers {
		var cfg struct {
			Type    string `json:"type"`
			Left    bool   `json:"strip_left"`
			Right   bool   `json:"strip_right"`
			Chars   string `json:"precompiled_charsmap"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
			Content string `json:"content"`
		}
		if err := json.Unmarshal(n, &cfg); err != nil {
			return nil, err
		}
		switch {
		case i == 0 && cfg.Type == "Strip" && cfg.Left && cfg.Right:
		case i == 1 && cfg.Type == "Precompiled":
			b, err := base64.StdEncoding.DecodeString(cfg.Chars)
			if err != nil || len(b) < 4 {
				return nil, fmt.Errorf("invalid precompiled normalization table")
			}
			n := int(binary.LittleEndian.Uint32(b))
			if n < 1024 || n%1024 != 0 || n >= len(b)-4 {
				return nil, fmt.Errorf("invalid precompiled normalization trie")
			}
			for j := 4; j < 4+n; j += 4 {
				u.chars = append(u.chars, binary.LittleEndian.Uint32(b[j:]))
			}
			u.replacements = string(b[4+n:])
		case i == 2 && cfg.Type == "Replace" && cfg.Pattern.Regex == " {2,}" && cfg.Content == " ":
		default:
			return nil, fmt.Errorf("unsupported Unigram normalizer step %d (%s)", i, cfg.Type)
		}
	}
	var pre struct {
		Type  string `json:"type"`
		Items []struct {
			Type        string `json:"type"`
			Replacement string `json:"replacement"`
			Prepend     string `json:"prepend_scheme"`
			Split       bool   `json:"split"`
		} `json:"pretokenizers"`
	}
	if err := json.Unmarshal(raw.PreTokenizer, &pre); err != nil {
		return nil, err
	}
	if pre.Type != "Sequence" || len(pre.Items) != 1 || pre.Items[0].Type != "Metaspace" || pre.Items[0].Replacement != "▁" || pre.Items[0].Prepend != "always" || !pre.Items[0].Split {
		return nil, fmt.Errorf("unsupported Unigram pre-tokenizer")
	}
	return u, nil
}

func (u *Unigram) TokenID(token string) (int32, bool) {
	if id, ok := u.added[token]; ok {
		return id, true
	}
	id, ok := u.pieces[token]
	return id, ok
}

func (u *Unigram) VocabSize() int {
	n := len(u.scores)
	for _, id := range u.added {
		n = max(n, int(id)+1)
	}
	return n
}

func (u *Unigram) HasAddedToken(s string) bool {
	for token := range u.added {
		if strings.Contains(s, token) {
			return true
		}
	}
	return false
}

// normalize applies the longest matching rule in the SentencePiece Darts
// trie. Every lookup is bounded because the table comes from model data.
func (u *Unigram) normalize(s string) string {
	s = strings.TrimSpace(s)
	var out strings.Builder
	offset := func(v uint32) uint32 { return (v >> 10) << ((v & (1 << 9)) >> 6) }
	for len(s) > 0 {
		pos := offset(u.chars[0])
		n, value := 0, 0
		for i := 0; i < len(s); i++ {
			pos ^= uint32(s[i])
			if int(pos) >= len(u.chars) {
				break
			}
			unit := u.chars[pos]
			if unit&0x800000ff != uint32(s[i]) {
				break
			}
			pos ^= offset(unit)
			if int(pos) >= len(u.chars) {
				break
			}
			if unit&256 != 0 {
				n, value = i+1, int(u.chars[pos]&0x7fffffff)
			}
		}
		if n > 0 && value < len(u.replacements) {
			if end := strings.IndexByte(u.replacements[value:], 0); end >= 0 {
				out.WriteString(u.replacements[value : value+end])
				s = s[n:]
				continue
			}
		}
		_, n = utf8.DecodeRuneInString(s)
		out.WriteString(s[:n])
		s = s[n:]
	}
	return out.String()
}

// EncodeWord returns pieces for one word in an is_split_into_words input.
// Special markers are handled by the caller; user text cannot create markers.
func (u *Unigram) EncodeWord(s string) []int32 {
	var ids []int32
	for word := range strings.SplitSeq(u.normalize(s), " ") {
		if word == "" {
			continue
		}
		if !strings.HasPrefix(word, "▁") {
			word = "▁" + word
		}
		ids = append(ids, u.segment(word)...)
	}
	return ids
}

func (u *Unigram) segment(s string) []int32 {
	type path struct {
		score    float64
		previous int
		id       int32
	}
	best := make([]path, len(s)+1)
	for i := 1; i < len(best); i++ {
		best[i].score = math.Inf(-1)
	}
	for start := 0; start < len(s); {
		_, size := utf8.DecodeRuneInString(s[start:])
		hasRune := false
		for end := start + 1; end <= min(len(s), start+u.maxPiece); end++ {
			id, ok := u.pieces[s[start:end]]
			if !ok {
				continue
			}
			if end-start == size {
				hasRune = true
			}
			score := best[start].score + u.scores[id]
			if score > best[end].score {
				best[end] = path{score, start, id}
			}
		}
		if !hasRune {
			end := start + size
			score := best[start].score + u.unknownScore
			if score > best[end].score {
				best[end] = path{score, start, u.unknown}
			}
		}
		start += size
	}
	var ids []int32
	for i := len(s); i > 0; i = best[i].previous {
		id := best[i].id
		if id != u.unknown || len(ids) == 0 || ids[len(ids)-1] != id {
			ids = append(ids, id)
		}
	}
	slices.Reverse(ids)
	return ids
}
