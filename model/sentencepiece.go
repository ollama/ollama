package model

import (
	"cmp"
	"fmt"
	"log/slog"
	"strconv"
	"strings"

	"github.com/emirpasic/gods/v2/trees/binaryheap"
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

		runes := []rune(text)

		root := &merge{offset: len(runes) - 1, size: 1}
		for i := len(runes) - 2; i >= 0; i-- {
			m := &merge{offset: i, size: 1, next: root}
			root.prev = m
			root = m
		}

		pairwise := func(a, b *merge) *pair[float32] {
			if a != nil && b != nil {
				aa := string(runes[a.offset : a.offset+a.size])
				bb := string(runes[b.offset : b.offset+b.size])
				if id := spm.vocab.Encode(aa + bb); id >= 0 {
					return &pair[float32]{a: a, b: b, rank: spm.vocab.Scores[id]}
				}
			}

			return nil
		}

		pairs := binaryheap.NewWith(func(i, j *pair[float32]) int { return cmp.Compare(i.rank, j.rank) })
		for m := root; m != nil; m = m.next {
			if pair := pairwise(m, m.next); pair != nil {
				pairs.Push(pair)
			}
		}

		for !pairs.Empty() {
			p, _ := pairs.Pop()
			a := string(runes[p.a.offset : p.a.offset+p.a.size])
			b := string(runes[p.b.offset : p.b.offset+p.b.size])
			if id := spm.vocab.Encode(a + b); a == "" || b == "" || id < 0 || spm.vocab.Scores[id] != p.rank {
				continue
			}

			p.a.size += p.b.size
			p.b.size = 0

			p.a.next = p.b.next
			if p.b.next != nil {
				p.b.next.prev = p.a
			}

			if pair := pairwise(p.a.prev, p.a); pair != nil {
				pairs.Push(pair)
			}

			if pair := pairwise(p.a, p.a.next); pair != nil {
				pairs.Push(pair)
			}
		}

		for m := root; m != nil; m = m.next {
			if s := string(runes[m.offset : m.offset+m.size]); s != "" {
				if id := spm.vocab.Encode(s); id >= 0 {
					ids = append(ids, id)
					continue
				}

				var result []int32
				for _, b := range []byte(s) {
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
