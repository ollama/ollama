package phonon

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/cases"
	"golang.org/x/text/language"
	"golang.org/x/text/unicode/norm"
)

const metaspace = "▁"

// unigram is the SentencePiece Unigram tokenizer from tokenizer.json, with the
// Prepend("▁") normalizer, Metaspace pre-tokenizer and byte fallback the
// released checkpoints use.
type unigram struct {
	pieces     []string
	scores     []float64
	byteIDs    [256]int32 // token ID for each byte, or -1
	bytes      []int      // byte value of each <0xNN> piece, or -1
	ids        map[string]int32
	chars      map[rune]bool // runes with a piece of their own
	maxLen     int
	unkID      int32
	unkScore   float64
	specials   []specialToken
	specialIDs []bool
}

type specialToken struct {
	ID      int32  `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
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
	u := &unigram{ids: make(map[string]int32, len(t.Model.Vocab)), chars: map[rune]bool{}, unkID: t.Model.UnkID, specialIDs: make([]bool, len(t.Model.Vocab))}
	for i := range u.byteIDs {
		u.byteIDs[i] = -1
	}
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
				u.byteIDs[b] = int32(i)
			}
		}
		u.pieces = append(u.pieces, piece)
		u.scores = append(u.scores, score)
		u.bytes = append(u.bytes, b)
		u.ids[piece] = int32(i)
		if rest := strings.TrimPrefix(piece, metaspace); utf8.RuneCountInString(rest) == 1 {
			c, _ := utf8.DecodeRuneInString(rest)
			u.chars[c] = true
		}
		u.maxLen = max(u.maxLen, len(piece))
		minScore = min(minScore, score)
	}
	u.unkScore = minScore - 10
	for _, token := range t.AddedTokens {
		if token.Content == "" || token.ID < 0 || int(token.ID) >= len(u.pieces) {
			return nil, fmt.Errorf("invalid added token %q", token.Content)
		}
		u.specials = append(u.specials, token)
		u.specialIDs[token.ID] = token.Special
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
	var ids []int32
	for word := range strings.SplitSeq(strings.ReplaceAll(u.fold(text), " ", metaspace), metaspace) {
		ids = u.encodeWord(metaspace+word, ids)
	}
	return ids
}

// letterFolds spell the letters that do not decompose into a letter and marks.
var letterFolds = map[rune]string{
	'Æ': "Ae", 'æ': "ae", 'Œ': "Oe", 'œ': "oe", 'ß': "ss", 'ẞ': "SS", 'Ø': "O", 'ø': "o",
	'Ł': "L", 'ł': "l", 'Đ': "D", 'đ': "d", 'Ð': "D", 'ð': "d", 'Þ': "Th", 'þ': "th",
}

// fold spells the letters that have no piece with ones that do, so that they
// do not fall back to bytes: "École" becomes "école" when only "é" has a piece,
// "ÉCOLE" "ECOLE", "ï" "i", and "ß" "ss". Anything else is left to byte
// fallback.
func (u *unigram) fold(text string) string {
	missing := func(c rune) bool { return unicode.IsLetter(c) && !u.chars[c] }
	if !strings.ContainsFunc(text, missing) {
		return text
	}
	runes := []rune(text)
	var b strings.Builder
	for i, c := range runes {
		if !missing(c) {
			b.WriteRune(c)
			continue
		}
		// A capital next to another keeps its case.
		caps := unicode.IsUpper(c) && (i > 0 && unicode.IsUpper(runes[i-1]) || i+1 < len(runes) && unicode.IsUpper(runes[i+1]))
		s := letterFolds[c]
		if caps {
			s = strings.ToUpper(s)
		}
		if lower := unicode.ToLower(c); lower >= utf8.RuneSelf && u.chars[lower] && !caps {
			s = string(lower)
		} else if s == "" {
			s = strings.Map(func(r rune) rune {
				if unicode.Is(unicode.Mn, r) {
					return -1
				}
				return r
			}, norm.NFKD.String(string(c)))
		}
		if s == "" || strings.ContainsFunc(s, func(r rune) bool { return !u.chars[r] }) {
			s = string(c)
		}
		b.WriteString(s)
	}
	return b.String()
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

	start := len(ids)
	for j := n; j > 0; j = from[j] {
		if tok[j] != u.unkID {
			ids = append(ids, tok[j])
			continue
		}
		// Backtracking visits pieces and their fallback bytes in reverse.
		mark := len(ids)
		for k := j - 1; k >= from[j]; k-- {
			if id := u.byteIDs[s[k]]; id >= 0 {
				ids = append(ids, id)
			} else {
				ids = append(ids[:mark], u.unkID)
				break
			}
		}
	}
	slices.Reverse(ids[start:])
	return ids
}

func (u *unigram) decode(ids []int32) string {
	var b []byte
	first := true
	for _, id := range ids {
		if u.specialIDs[id] {
			continue
		}
		if c := u.bytes[id]; c >= 0 {
			b = append(b, byte(c))
			first = false
			continue
		}
		piece := u.pieces[id]
		if first {
			piece = strings.ReplaceAll(piece, metaspace, "")
		} else {
			piece = strings.ReplaceAll(piece, metaspace, " ")
		}
		b = append(b, piece...)
		first = false
	}
	return strings.ToValidUTF8(string(b), "�")
}

const maxChunkTokens = 50

var errEmptyText = errors.New("input has no text to speak")

// prepareText follows the reference prepare_text_prompt, after normalization.
func prepareText(text string) (string, int, error) {
	words := strings.Fields(text)
	text = strings.Join(words, " ")
	if text == "" {
		return "", 0, errEmptyText
	}
	framesAfterEOS := 1
	if len(words) <= 4 {
		framesAfterEOS = 3
	}
	// Rust's full case mapping: "ß" becomes "SS" and "ﬁ" "FI".
	_, n := utf8.DecodeRuneInString(text)
	text = cases.Upper(language.Und).String(text[:n]) + text[n:]
	if last, _ := utf8.DecodeLastRuneInString(text); isAlphanumeric(last) {
		text += "."
	}
	return text, framesAfterEOS, nil
}

// chunks splits text into sentence groups of at most maxChunkTokens tokens,
// as Phonon split_into_best_sentences does, then fits oversized sentences
// to the reference runtime's hard limit.
func (u *unigram) chunks(text string) ([]string, error) {
	text, _, err := prepareText(normalizeText(text))
	if err != nil {
		return nil, err
	}
	tokens := u.encode(strings.TrimSpace(text))

	type segment struct {
		n    int
		text string
	}
	boundary := map[int32]bool{}
	for _, id := range u.encode(".!...?")[1:] {
		boundary[id] = true
	}
	var segs []segment
	start, after := 0, false
	for i, id := range tokens {
		if boundary[id] {
			after = true
			continue
		}
		if after {
			segs = append(segs, segment{i - start, u.decode(tokens[start:i])})
			start = i
		}
		after = false
	}
	segs = append(segs, segment{len(tokens) - start, u.decode(tokens[start:])})

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
	return u.fitChunks(chunks)
}

// fitChunks mirrors upstream plan::fit_or_error. Long prompts otherwise outlast
// the model's training segments and can lose words even with enough KV space.
func (u *unigram) fitChunks(chunks []string) ([]string, error) {
	const limit = 200
	slices.Reverse(chunks)
	var fitted []string
	for len(chunks) > 0 {
		text := chunks[len(chunks)-1]
		chunks = chunks[:len(chunks)-1]
		prepared, _, err := prepareText(text)
		if err != nil {
			return nil, err
		}
		if len(u.encode(prepared)) <= limit {
			fitted = append(fitted, text)
			continue
		}
		words := strings.Fields(text)
		var left, right string
		if len(words) > 1 {
			cut := u.cutWords(words, limit/4)
			left, right = strings.Join(words[:cut], " "), strings.Join(words[cut:], " ")
		} else {
			runes := []rune(text)
			if len(runes) < 2 {
				return nil, fmt.Errorf("text cannot fit the %d-token limit", limit)
			}
			left, right = string(runes[:len(runes)/2]), string(runes[len(runes)/2:])
		}
		chunks = append(chunks, right, left)
	}
	return fitted, nil
}

// Prefer a sentence end, then a clause end, then the most balanced word split.
func (u *unigram) cutWords(words []string, minSide int) int {
	ends := make([]int, len(words))
	total := 0
	for i, word := range words {
		total += len(u.encode(word))
		ends[i] = total
	}
	for _, marks := range []string{".!?", ",;:", ""} {
		best, distance := 0, total+1
		for i, end := range ends[:len(ends)-1] {
			if marks != "" {
				word := strings.TrimRight(words[i], "\"')]”’»")
				last, _ := utf8.DecodeLastRuneInString(word)
				if !strings.ContainsRune(marks, last) || end < minSide || total-end < minSide {
					continue
				}
			}
			d := end - (total - end)
			if d < 0 {
				d = -d
			}
			if d < distance {
				best, distance = i+1, d
			}
		}
		if best != 0 {
			return best
		}
	}
	return 1
}
