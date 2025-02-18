package model

import (
	"iter"
	"log/slog"
	"strings"

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
	slog.Debug("Tokens", "num tokens", len(vocab.Values), "vals", vocab.Values[:3], "scores", vocab.Scores[:3], "types", vocab.Types[:3])

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

	slog.Debug("Token counts", "normal", counter[TOKEN_TYPE_NORMAL], "unknown", counter[TOKEN_TYPE_UNKNOWN], "control", counter[TOKEN_TYPE_CONTROL],
		"user defined", counter[TOKEN_TYPE_USER_DEFINED], "unused", counter[TOKEN_TYPE_UNUSED], "byte", counter[TOKEN_TYPE_BYTE],
		"max token len", maxTokenLen)

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
	slog.Debug("fragments", "frags", fragments)

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

			pairwise := func(a, b int) *candidate {
				if a < 0 || b >= len(runes) {
					return nil
				}

				left, right := string(merges[a].runes), string(merges[b].runes)
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
				slog.Debug("candidate", "candidate", e)
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

			slog.Debug("merges", "merges", merges)

			for _, merge := range merges {
				if len(merge.runes) > 0 {
					if id := spm.vocab.Encode(string(merge.runes)); id >= 0 {
						ids = append(ids, id)
					} else {
						slog.Debug("missing token", "token", string(merge.runes))
					}
				}
			}
		}
	}
	slog.Debug("encoded", "ids", ids)

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
