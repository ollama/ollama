package model

import (
	"container/heap"
	"fmt"
	"log/slog"
	"strconv"
	"strings"

	"github.com/ollama/ollama/logutil"
)

const spmWhitespaceSep = "▁"

type SentencePiece struct {
	maxTokenLen int
	vocab       *Vocabulary
}

var _ TextProcessor = (*SentencePiece)(nil)

func (spm SentencePiece) Vocabulary() *Vocabulary {
	return spm.vocab
}

func NewSentencePiece(vocab *Vocabulary) SentencePiece {
	logutil.Trace("Tokens", "num tokens", len(vocab.Values), "vals", vocab.Values[:5], "scores", vocab.Scores[:5], "types", vocab.Types[:5])

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

		if id := spm.vocab.Encode(text); id >= 0 {
			ids = append(ids, id)
			continue
		}

		q := &queue{}
		heap.Init(q)

		runes := []rune(text)
		merges := make([]merge, len(runes))
		for r := range runes {
			merges[r] = merge{
				p:     r - 1,
				n:     r + 1,
				runes: []rune{runes[r]},
			}
		}

		pairwise := func(a, b int) *candidate {
			if a < 0 || b >= len(runes) {
				return nil
			}

			left, right := string(merges[a].runes), string(merges[b].runes)
			if id := spm.vocab.Encode(left + right); id >= 0 {
				return &candidate{
					a:     a,
					b:     b,
					score: spm.vocab.Scores[id],
					size:  len(left) + len(right),
				}
			}

			return nil
		}

		for i := range len(runes) - 1 {
			if pair := pairwise(i, i+1); pair != nil {
				heap.Push(q, pair)
			}
		}

		for q.Len() > 0 {
			pair := heap.Pop(q).(*candidate)
			left, right := merges[pair.a], merges[pair.b]

			if string(left.runes) == "" || string(right.runes) == "" || len(string(left.runes))+len(string(right.runes)) != pair.size {
				continue
			}

			merges[pair.a].runes = append(left.runes, right.runes...)
			merges[pair.b].runes = nil
			merges[pair.a].n = right.n
			if right.n < len(merges) {
				merges[right.n].p = pair.a
			}

			if pair := pairwise(merges[pair.a].p, pair.a); pair != nil {
				heap.Push(q, pair)
			}

			if pair := pairwise(pair.a, merges[pair.a].n); pair != nil {
				heap.Push(q, pair)
			}
		}

		for _, merge := range merges {
			if token := string(merge.runes); token != "" {
				id := spm.vocab.Encode(token)

				if id >= 0 {
					ids = append(ids, id)
					continue
				}

				// Fallback to byte tokenization
				var result []int32
				for _, b := range []byte(token) {
					byteToken := fmt.Sprintf("<0x%02X>", b)
					unknownID := spm.vocab.Encode(byteToken)
					if unknownID >= 0 {
						result = append(result, unknownID)
					} else {
						slog.Debug("unknown byte token", "byte", b, "token", byteToken)
					}
				}

				ids = append(ids, result...)
			}
		}
	}

	if addSpecial {
		ids = spm.vocab.addSpecials(ids)
	}

	logutil.Trace("encoded", "string", s, "ids", ids)
	return ids, nil
}

type candidate struct {
	a, b  int
	score float32
	size  int
}

type queue []*candidate

func (q queue) Len() int { return len(q) }

func (q queue) Less(i, j int) bool {
	return (q[i].score > q[j].score) || (q[i].score == q[j].score && q[i].a < q[j].a)
}

func (q queue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }

func (q *queue) Push(x interface{}) {
	item := x.(*candidate)
	*q = append(*q, item)
}

func (q *queue) Pop() interface{} {
	old := *q
	n := len(old)
	item := old[n-1]
	*q = old[0 : n-1]
	return item
}

func (spm SentencePiece) Decode(ids []int32) (string, error) {
	var sb strings.Builder
	for _, id := range ids {
		data := spm.vocab.Decode(id)
		data = strings.ReplaceAll(data, spmWhitespaceSep, " ")

		// For tokenizers that use byte tokens like "<0xEA>"
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
