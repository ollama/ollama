package tokenizer

import (
	"fmt"
	"log/slog"
	"math"
	"strconv"
	"strings"

	"github.com/ollama/ollama/logutil"
)

const spmWhitespaceSep = "▁"

type SentencePiece struct {
	maxTokenLen int
	vocab       *Vocabulary
}

var _ Tokenizer = (*SentencePiece)(nil)

func (spm SentencePiece) Vocabulary() *Vocabulary {
	return spm.vocab
}

func NewSentencePiece(vocab *Vocabulary) SentencePiece {
	end := min(5, len(vocab.Values))
	logutil.Trace("Tokens", "num tokens", len(vocab.Values), "vals", vocab.Values[:end], "scores", vocab.Scores[:end], "types", vocab.Types[:end])

	counter := map[int]int{}
	var maxTokenLen int
	for cnt := range vocab.Types {
		switch vocab.Types[cnt] {
		case TOKEN_TYPE_NORMAL, TOKEN_TYPE_USER_DEFINED, TOKEN_TYPE_UNUSED:
			maxTokenLen = max(maxTokenLen, len(vocab.Values[cnt]))
			fallthrough
		default:
			counter[int(vocab.Types[cnt])] += 1
		}
	}

	logutil.Trace("Token counts", "normal", counter[TOKEN_TYPE_NORMAL], "unknown", counter[TOKEN_TYPE_UNKNOWN], "control", counter[TOKEN_TYPE_CONTROL],
		"user defined", counter[TOKEN_TYPE_USER_DEFINED], "unused", counter[TOKEN_TYPE_UNUSED], "byte", counter[TOKEN_TYPE_BYTE],
		"max token len", maxTokenLen)

	return SentencePiece{
		maxTokenLen: maxTokenLen,
		vocab:       vocab,
	}
}

func (spm SentencePiece) Is(id int32, special Special) bool {
	return spm.vocab.Is(id, special)
}

func (spm SentencePiece) Encode(s string, addSpecial bool) ([]int32, error) {
	if spm.vocab.AddSpacePrefix {
		s = " " + s
	}
	fragments := []fragment{{value: s}}
	for _, special := range spm.vocab.SpecialVocabulary() {
		id := spm.vocab.Encode(special)
		for i := 0; i < len(fragments); i++ {
			frag := fragments[i]
			if len(frag.ids) > 0 {
				continue
			}

			var middle []fragment
			switch i := strings.Index(frag.value, special); {
			case i < 0:
				middle = append(middle, frag)
			case i > 0:
				middle = append(middle, fragment{value: frag.value[:i]})
				fallthrough
			default:
				middle = append(middle, fragment{value: special, ids: []int32{id}})
				if rest := frag.value[i+len(special):]; rest != "" {
					middle = append(middle, fragment{value: rest})
				}
			}

			fragments = append(fragments[:i], append(middle, fragments[i+1:]...)...)
		}
	}

	var ids []int32
	for _, frag := range fragments {
		if len(frag.ids) > 0 {
			ids = append(ids, frag.ids...)
			continue
		}

		text := strings.ReplaceAll(frag.value, " ", spmWhitespaceSep)

		ids = append(ids, spm.tokenizeViterbi(text)...)
	}

	if addSpecial {
		ids = spm.vocab.addSpecials(ids)
	}

	logutil.Trace("encoded", "string", s, "ids", ids)
	return ids, nil
}

// tokenizeViterbi segments text into vocabulary tokens using the Viterbi algorithm,
// finding the globally optimal (highest log-probability) segmentation.
func (spm SentencePiece) tokenizeViterbi(text string) []int32 {
	runes := []rune(text)
	n := len(runes)
	if n == 0 {
		return nil
	}

	// dp[i] = best cumulative score for segmenting runes[0:i]
	dp := make([]float32, n+1)
	for i := range dp {
		dp[i] = float32(math.Inf(-1))
	}
	dp[0] = 0

	// back[i] = rune-length of the token ending at position i in the best path
	back := make([]int, n+1)

	for i := 0; i < n; i++ {
		if math.IsInf(float64(dp[i]), -1) {
			continue
		}
		for l := 1; i+l <= n && l <= spm.maxTokenLen; l++ {
			piece := string(runes[i : i+l])
			id := spm.vocab.Encode(piece)
			if id < 0 {
				continue
			}
			score := dp[i] + spm.vocab.Scores[id]
			if score > dp[i+l] {
				dp[i+l] = score
				back[i+l] = l
			}
		}
	}

	// Traceback from position n to 0
	var ids []int32
	pos := n
	for pos > 0 {
		l := back[pos]
		if l == 0 {
			// Position unreachable via vocab — fall back to byte tokens for this char
			ch := string(runes[pos-1 : pos])
			var result []int32
			for _, b := range []byte(ch) {
				byteToken := fmt.Sprintf("<0x%02X>", b)
				uid := spm.vocab.Encode(byteToken)
				if uid >= 0 {
					result = append(result, uid)
				} else {
					slog.Debug("unknown byte token", "byte", b, "token", byteToken)
				}
			}
			// Prepend in reverse order since we're tracing backwards
			for i := len(result) - 1; i >= 0; i-- {
				ids = append(ids, result[i])
			}
			pos--
			continue
		}
		piece := string(runes[pos-l : pos])
		id := spm.vocab.Encode(piece)
		if id >= 0 {
			ids = append(ids, id)
		}
		pos -= l
	}

	// Reverse: traceback produces tokens in reverse order
	for i, j := 0, len(ids)-1; i < j; i, j = i+1, j-1 {
		ids[i], ids[j] = ids[j], ids[i]
	}

	return ids
}

func (spm SentencePiece) Decode(ids []int32) (string, error) {
	var sb strings.Builder
	for _, id := range ids {
		data := spm.vocab.Decode(id)
		data = strings.ReplaceAll(data, spmWhitespaceSep, " ")

		// For tokenizer that use byte tokens like "<0xEA>"
		// convert them to the partial unicode character
		// so they are buffered correctly by the runner instead
		// of being sent back to the api as "<0xEA>"
		if len(data) == 6 && strings.HasPrefix(data, "<0x") && strings.HasSuffix(data, ">") {
			byteVal, err := strconv.ParseUint(data[1:5], 0, 8)
			if err != nil {
				return "", fmt.Errorf("failed to parse hex byte: %v", err)
			}

			if err := sb.WriteByte(byte(byteVal)); err != nil {
				return "", err
			}
		} else {
			if _, err := sb.WriteString(data); err != nil {
				return "", err
			}
		}
	}

	logutil.Trace("decoded", "ids", ids, "string", sb.String())
	return sb.String(), nil
}
