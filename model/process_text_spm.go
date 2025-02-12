package model

import (
	"fmt"
	"iter"
	"log/slog"
	"strings"
	//"unicode/utf8"

	"github.com/dlclark/regexp2"
	queue "github.com/emirpasic/gods/queues/priorityqueue"
)

const spmWhitespaceSep = "▁"

func replaceWhitespaceBySeperator(s string) string {
	return strings.ReplaceAll(s, " ", spmWhitespaceSep)
}

type SentencePieceModel struct {
	maxTokenLen int
	pre         *regexp2.Regexp
	vocab       *Vocabulary
}

func NewSentencePieceModel(pre string, vocab *Vocabulary) SentencePieceModel {
	fmt.Printf("Tokens (%d): %5s %5s %5s ...\n", len(vocab.Values), vocab.Values[0], vocab.Values[1], vocab.Values[2])
	fmt.Printf("Scores (%d): %0.3f %0.3f %0.3f ...\n", len(vocab.Scores), vocab.Scores[0], vocab.Scores[1], vocab.Scores[2])
	fmt.Printf("Types  (%d): %5d %5d %5d ...\n", len(vocab.Types), vocab.Types[0], vocab.Types[1], vocab.Types[2])

	counter := map[int]int{}
	var maxTokenLen int

	for cnt, _ := range vocab.Types {
		switch vocab.Types[cnt] {
		case TOKEN_TYPE_NORMAL, TOKEN_TYPE_USER_DEFINED, TOKEN_TYPE_UNUSED:
			maxTokenLen = max(maxTokenLen, len(vocab.Values[cnt]))
			fallthrough
		default:
			counter[int(vocab.Types[cnt])] += 1
		}
	}

	fmt.Printf("Normal: %d\n", counter[TOKEN_TYPE_NORMAL])
	fmt.Printf("Unknown: %d\n", counter[TOKEN_TYPE_UNKNOWN])
	fmt.Printf("Control: %d\n", counter[TOKEN_TYPE_CONTROL])
	fmt.Printf("User Defined: %d\n", counter[TOKEN_TYPE_USER_DEFINED])
	fmt.Printf("Unused: %d\n", counter[TOKEN_TYPE_UNUSED])
	fmt.Printf("Byte: %d\n", counter[TOKEN_TYPE_BYTE])
	fmt.Printf("Max token len: %d\n", maxTokenLen)

	return SentencePieceModel{
		maxTokenLen: maxTokenLen,
		pre:         regexp2.MustCompile(pre, regexp2.Unicode|regexp2.RE2),
		vocab:       vocab,
	}
}

func (spm SentencePieceModel) Is(id int32, special Special) bool {
	return spm.vocab.Is(id, special)
}

func (spm *SentencePieceModel) split(s string) iter.Seq[string] {
	return func(yield func(string) bool) {
		for m, _ := spm.pre.FindStringMatch(s); m != nil; m, _ = spm.pre.FindNextMatch(m) {
			if !yield(m.String()) {
				break
			}
		}
	}
}

func (spm SentencePieceModel) Encode(s string) ([]int32, error) {
	fragments := []fragment{{value: s}}
	for _, special := range spm.vocab.SpecialVocabulary() {
		// TODO: process special tokens concurrently
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
	fmt.Printf("frags = %#v\n", fragments)

	var ids []int32
	for _, frag := range fragments {
		if len(frag.ids) > 0 {
			ids = append(ids, frag.ids...)
			continue
		}

		for split := range spm.split(frag.value) {
			split = replaceWhitespaceBySeperator(split)

			var sb strings.Builder
			sb.Write([]byte(split))
			if id := spm.vocab.Encode(sb.String()); id >= 0 {
				ids = append(ids, id)
				continue
			}

			runes := []rune(sb.String())
			pq := queue.NewWith(func(a, b any) int {
				priA := a.(*candidate)
				priB := b.(*candidate)
				if priA.score > priB.score || (priA.score == priB.score && priA.a < priB.a) {
					return 1
				}
				return -1
			})

			merges := make([]merge, len(runes))
			for r := range runes {
				merges[r] = merge{
					p:     r - 1,
					n:     r + 1,
					runes: []rune{runes[r]},
				}
			}
			fmt.Printf("remaining runes = %#v\n", runes)
			fmt.Printf("merges = %#v\n", merges)

			pairwise := func(a, b int) *candidate {
				if a < 0 || b >= len(runes) {
					return nil
				}

				left, right := string(merges[a].runes), string(merges[b].runes)
				fmt.Printf("looking up '%s'\n", left+right)
				if id := spm.vocab.Encode(left + right); id >= 0 {
					return &candidate{
						a:      a,
						b:      b,
						length: len(left + " " + right),
						score:  spm.vocab.Scores[id],
					}
				}
				return nil
			}

			for i := range len(runes) - 1 {
				if pair := pairwise(i, i+1); pair != nil {
					pq.Enqueue(pair)
				}
			}

			pqv := pq.Values()
			for _, v := range pqv {
				e := v.(*candidate)
				fmt.Printf("candidate = %#v\n", e)
			}

			for !pq.Empty() {
				v, _ := pq.Dequeue()
				pair := v.(*candidate)
				left, right := merges[pair.a], merges[pair.b]

				if len(left.runes) == 0 || len(right.runes) == 0 {
					continue
				}

				merges[pair.a].runes = append(left.runes, right.runes...)
				merges[pair.b].runes = nil
				merges[pair.a].n = right.n
				if right.n < len(merges) {
					merges[right.n].p = pair.a
				}

				if pair := pairwise(merges[pair.a].p, pair.a); pair != nil {
					pq.Enqueue(pair)
				}

				if pair := pairwise(pair.a, merges[pair.a].n); pair != nil {
					pq.Enqueue(pair)
				}
			}

			fmt.Printf("merges = %#v\n", merges)

			for _, merge := range merges {
				if len(merge.runes) > 0 {
					if id := spm.vocab.Encode(string(merge.runes)); id >= 0 {
						ids = append(ids, id)
					} else {
						fmt.Printf("!!! missing token for '%s'\n", string(merge.runes))
					}
				}
			}
		}

	}
	fmt.Printf("tokens = %#v\n", ids)

	return ids, nil
}

type candidate struct {
	a, b   int
	score  float32
	length int
}

func (spm SentencePieceModel) Decode(ids []int32) (string, error) {
	var sb strings.Builder
	for _, id := range ids {
		data := spm.vocab.Decode(id)
		data = strings.ReplaceAll(data, spmWhitespaceSep, " ")
		if _, err := sb.WriteString(data); err != nil {
			return "", err
		}
	}

	slog.Debug("decoded", "ids", ids, "text", sb.String())
	return sb.String(), nil
}
