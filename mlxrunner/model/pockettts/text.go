package pockettts

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

const metaspace = "▁"

// unigram is the SentencePiece Unigram tokenizer from tokenizer.json, with the
// Prepend("▁") normalizer, Metaspace pre-tokenizer and byte fallback the
// released checkpoints use.
type unigram struct {
	pieces   []string
	scores   []float64
	bytes    []int // byte value of each <0xNN> piece, or -1
	ids      map[string]int32
	maxLen   int
	unkID    int32
	unkScore float64
	specials []specialToken
}

type specialToken struct {
	ID      int32  `json:"id"`
	Content string `json:"content"`
}

func newUnigram(data []byte) (*unigram, error) {
	var t struct {
		AddedTokens []specialToken `json:"added_tokens"`
		Model       struct {
			Type  string  `json:"type"`
			UnkID int32   `json:"unk_id"`
			Vocab [][]any `json:"vocab"`
		} `json:"model"`
	}
	if err := json.Unmarshal(data, &t); err != nil {
		return nil, err
	}
	if t.Model.Type != "Unigram" {
		return nil, fmt.Errorf("unsupported tokenizer model %q", t.Model.Type)
	}
	u := &unigram{ids: make(map[string]int32, len(t.Model.Vocab)), unkID: t.Model.UnkID}
	minScore := math.Inf(1)
	if len(t.Model.Vocab) == 0 || t.Model.UnkID < 0 || int(t.Model.UnkID) >= len(t.Model.Vocab) {
		return nil, fmt.Errorf("invalid unigram vocabulary or unknown token")
	}
	for i, v := range t.Model.Vocab {
		if len(v) != 2 {
			return nil, fmt.Errorf("invalid vocab entry %d", i)
		}
		piece, ok1 := v[0].(string)
		score, ok2 := v[1].(float64)
		if !ok1 || !ok2 || piece == "" {
			return nil, fmt.Errorf("invalid vocab entry %d", i)
		}
		b := -1
		if len(piece) == 6 && strings.HasPrefix(piece, "<0x") && piece[5] == '>' {
			if v, err := strconv.ParseUint(piece[3:5], 16, 8); err == nil {
				b = int(v)
			}
		}
		u.pieces = append(u.pieces, piece)
		u.scores = append(u.scores, score)
		u.bytes = append(u.bytes, b)
		u.ids[piece] = int32(i)
		u.maxLen = max(u.maxLen, len(piece))
		minScore = min(minScore, score)
	}
	u.unkScore = minScore - 10
	for _, token := range t.AddedTokens {
		if token.Content == "" || token.ID < 0 || int(token.ID) >= len(u.pieces) {
			return nil, fmt.Errorf("invalid added token %q", token.Content)
		}
		u.specials = append(u.specials, token)
	}
	return u, nil
}

func (u *unigram) encode(text string) []int32 {
	var ids []int32
	for text != "" {
		at, selected := len(text), -1
		for i, token := range u.specials {
			if pos := strings.Index(text, token.Content); pos >= 0 && (pos < at ||
				(pos == at && selected >= 0 && len(token.Content) > len(u.specials[selected].Content))) {
				at, selected = pos, i
			}
		}
		if at > 0 {
			ids = append(ids, u.encodeText(text[:at])...)
		}
		if selected < 0 {
			break
		}
		token := u.specials[selected]
		ids = append(ids, token.ID)
		text = text[at+len(token.Content):]
	}
	return ids
}

func (u *unigram) encodeText(text string) []int32 {
	s := metaspace + strings.ReplaceAll(text, " ", metaspace)
	var ids []int32
	start := 0
	for i := len(metaspace); i < len(s); {
		if strings.HasPrefix(s[i:], metaspace) {
			ids = u.encodeWord(s[start:i], ids)
			start = i
			i += len(metaspace)
			continue
		}
		_, n := utf8.DecodeRuneInString(s[i:])
		i += n
	}
	return u.encodeWord(s[start:], ids)
}

// encodeWord appends the Viterbi segmentation of s.
func (u *unigram) encodeWord(s string, ids []int32) []int32 {
	n := len(s)
	best := make([]float64, n+1)
	from := make([]int, n+1)
	tok := make([]int32, n+1)
	for i := range from {
		from[i] = -1
	}
	for i := 0; i < n; {
		_, size := utf8.DecodeRuneInString(s[i:])
		single := false
		for j := i + 1; j <= min(n, i+u.maxLen); j++ {
			id, ok := u.ids[s[i:j]]
			if !ok {
				continue
			}
			if score := best[i] + u.scores[id]; from[j] < 0 || score > best[j] {
				best[j], from[j], tok[j] = score, i, id
			}
			single = single || j-i == size
		}
		if j := i + size; !single {
			if score := best[i] + u.unkScore; from[j] < 0 || score > best[j] {
				best[j], from[j], tok[j] = score, i, u.unkID
			}
		}
		i += size
	}

	var rev []int32
	for j := n; j > 0; j = from[j] {
		if tok[j] != u.unkID {
			rev = append(rev, tok[j])
			continue
		}
		piece := s[from[j]:j]
		bytes := make([]int32, 0, len(piece))
		for k := range len(piece) {
			if id, ok := u.ids[fmt.Sprintf("<0x%02X>", piece[k])]; ok {
				bytes = append(bytes, id)
			}
		}
		if len(bytes) != len(piece) {
			bytes = []int32{u.unkID}
		}
		for k := len(bytes) - 1; k >= 0; k-- {
			rev = append(rev, bytes[k])
		}
	}
	for k := len(rev) - 1; k >= 0; k-- {
		ids = append(ids, rev[k])
	}
	return ids
}

func (u *unigram) decode(ids []int32) string {
	var b []byte
	for i, id := range ids {
		if c := u.bytes[id]; c >= 0 {
			b = append(b, byte(c))
			continue
		}
		piece := u.pieces[id]
		if i == 0 {
			piece = strings.ReplaceAll(piece, metaspace, "")
		} else {
			piece = strings.ReplaceAll(piece, metaspace, " ")
		}
		b = append(b, piece...)
	}
	return strings.ToValidUTF8(string(b), "�")
}

const maxChunkTokens = 50

var errEmptyText = errors.New("input text is empty")

// textOptions are the per-language text rewrites in the upstream config.
type textOptions struct {
	PadShortInputs    bool              `json:"pad_with_spaces_for_short_inputs"`
	RemoveSemicolons  bool              `json:"remove_semicolons"`
	ReplaceCharacters map[string]string `json:"replace_characters"`
}

var punctuationAfterSentence = regexp.MustCompile(`([.!?\x{2026}])\s*[,;:]`)

// prepare mirrors pocket_tts.prepare_text_prompt. It returns the text and the
// frames to keep after EOS.
func (o textOptions) prepare(text string) (string, int, error) {
	text = strings.TrimSpace(text)
	if len(o.ReplaceCharacters) > 0 {
		var b strings.Builder
		for _, r := range text {
			if v, ok := o.ReplaceCharacters[string(r)]; ok {
				b.WriteString(v)
			} else {
				b.WriteRune(r)
			}
		}
		text = punctuationAfterSentence.ReplaceAllString(strings.Join(strings.Fields(b.String()), " "), "$1")
	}
	if text == "" {
		return "", 0, errEmptyText
	}
	text = strings.NewReplacer("\n", " ", "\r", " ").Replace(text)
	text = strings.ReplaceAll(text, "  ", " ")
	if o.RemoveSemicolons {
		text = strings.ReplaceAll(text, ";", ",")
	}

	framesAfterEOS := 1
	if len(strings.Fields(text)) <= 4 {
		framesAfterEOS = 3
	}

	if r, n := utf8.DecodeRuneInString(text); !unicode.IsUpper(r) {
		text = string(unicode.ToUpper(r)) + text[n:]
	}

	const terminal, weak, closers = ".!?…", ",;:-–—", "\"'”’)]»"
	core := strings.TrimRight(text, closers+" ")
	if core != "" {
		last, _ := utf8.DecodeLastRuneInString(core)
		switch {
		case strings.ContainsRune(terminal, last):
		case strings.ContainsRune(weak, last):
			text = strings.TrimRight(core, weak+" ") + "." + strings.TrimSpace(text[len(core):])
		default:
			text += "."
		}
	}
	if o.PadShortInputs && len(strings.Fields(text)) < 5 {
		text = strings.Repeat(" ", 8) + text
	}
	return text, framesAfterEOS, nil
}

// chunks splits text into sentence groups of at most maxChunkTokens tokens,
// as pocket_tts.split_into_best_sentences does.
func (u *unigram) chunks(text string, opts textOptions) ([]string, error) {
	text, _, err := opts.prepare(text)
	if err != nil {
		return nil, err
	}
	tokens := u.encode(strings.TrimSpace(text))

	type segment struct {
		n    int
		text string
	}
	split := func(tokens []int32, delimiters string, skipDecimal bool) []segment {
		boundary := map[int32]bool{}
		for _, id := range u.encode(delimiters)[1:] {
			boundary[id] = true
		}
		var segs []segment
		start, after := 0, false
		for i, id := range tokens {
			if boundary[id] {
				after = true
				continue
			}
			if after && !(skipDecimal && u.decimalPoint(tokens, i)) {
				segs = append(segs, segment{i - start, u.decode(tokens[start:i])})
				start = i
			}
			after = false
		}
		return append(segs, segment{len(tokens) - start, u.decode(tokens[start:])})
	}

	var segs []segment
	for _, s := range split(tokens, ".!...?", true) {
		if s.n > maxChunkTokens {
			if sub := split(u.encode(strings.TrimSpace(s.text)), ",;:", false); len(sub) > 1 {
				segs = append(segs, sub...)
				continue
			}
		}
		segs = append(segs, s)
	}

	var chunks []string
	var cur string
	var n int
	for _, s := range segs {
		switch {
		case cur == "":
			cur, n = s.text, s.n
		case n+s.n > maxChunkTokens:
			chunks = append(chunks, strings.TrimSpace(cur))
			cur, n = s.text, s.n
		default:
			cur += " " + s.text
			n += s.n
		}
	}
	if cur != "" {
		chunks = append(chunks, strings.TrimSpace(cur))
	}
	return chunks, nil
}

func (u *unigram) decimalPoint(tokens []int32, i int) bool {
	prefix, suffix := u.decode(tokens[:i]), u.decode(tokens[i:])
	return len(prefix) >= 2 && prefix[len(prefix)-1] == '.' && isDigit(prefix[len(prefix)-2]) &&
		suffix != "" && isDigit(suffix[0])
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }
