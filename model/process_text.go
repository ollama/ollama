package model

import (
	"cmp"
	"log/slog"
	"strings"
	"sync"

	"github.com/dlclark/regexp2"
	heap "github.com/emirpasic/gods/v2/trees/binaryheap"
)

type Special int32

const (
	SpecialBOS Special = iota
	SpecialEOS
)

type TextProcessor interface {
	Encode(string) ([]int32, error)
	Decode([]int32) (string, error)
	Is(uint32, Special) bool
}

type Vocabulary struct {
	Values []string
	Types  []uint32
	Scores []uint32
	Merges []string

	BOS, EOS uint32

	specialOnce sync.Once
	special     []string

	valuesOnce sync.Once
	values     map[string]int32

	mergeOnce sync.Once
	merge     map[string]int32
}

func (v *Vocabulary) Is(id uint32, special Special) bool {
	switch special {
	case SpecialBOS:
		return id == v.BOS
	case SpecialEOS:
		return id == v.EOS
	default:
		return false
	}
}

func (v *Vocabulary) Encode(s string) int32 {
	v.valuesOnce.Do(func() {
		v.values = make(map[string]int32, len(v.Values))
		for i, value := range v.Values {
			v.values[value] = int32(i)
		}
	})

	if id, ok := v.values[s]; ok {
		return id
	}

	return -1
}

func (v *Vocabulary) Decode(id int32) string {
	return v.Values[id]
}

func (v *Vocabulary) SpecialVocabulary() []string {
	v.specialOnce.Do(func() {
		for i := range v.Values {
			if v.Types[i] == 3 {
				v.special = append(v.special, v.Values[i])
			}
		}
	})

	return v.special
}

func (v *Vocabulary) Merge(left, right string) int {
	v.mergeOnce.Do(func() {
		v.merge = make(map[string]int32, len(v.Merges))
		for i, merge := range v.Merges {
			v.merge[merge] = int32(i)
		}
	})

	if id, ok := v.merge[left+" "+right]; ok {
		return int(id)
	}

	return -1
}

type BytePairEncoding struct {
	Pretokenizer string

	*Vocabulary
}

func (bpe BytePairEncoding) split(s string) ([]string, error) {
	re, err := regexp2.Compile(bpe.Pretokenizer, regexp2.Unicode|regexp2.RE2)
	if err != nil {
		return nil, err
	}

	var matches []string
	for m, _ := re.FindStringMatch(s); m != nil; m, _ = re.FindNextMatch(m) {
		matches = append(matches, m.String())
	}

	return matches, nil
}

// fragment is a string fragment and their corresponding token IDs
type fragment struct {
	value string
	ids   []int32
}

// pair is a pair of runes and its rank
type pair struct {
	a, b  int
	rank  int
	value string
}

type merge struct {
	p, n  int
	runes []rune
}

func (bpe BytePairEncoding) Encode(s string) ([]int32, error) {
	fragments := []fragment{{value: s}}
	for _, special := range bpe.Vocabulary.SpecialVocabulary() {
		// TODO: process special tokens concurrently
		id := bpe.Vocabulary.Encode(special)
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
			slog.Debug("encoded", "text", frag.value, "ids", frag.ids, "special", true)
			continue
		}

		// split fragment using pretokenizer
		splits, err := bpe.split(frag.value)
		if err != nil {
			return nil, err
		}

		for _, split := range splits {
			// TODO: process splits concurrently
			var sb strings.Builder
			for _, b := range []byte(split) {
				r := rune(b)
				switch {
				case r == 0x00ad:
					r = 0x0143
				case r <= 0x0020:
					r = r + 0x0100
				case r >= 0x007e && r <= 0x00a0:
					r = r + 0x00a2
				}

				sb.WriteRune(r)
			}

			// short circuit if the fragment is in the vocabulary
			if id := bpe.Vocabulary.Encode(sb.String()); id >= 0 {
				ids = append(ids, id)
				slog.Debug("encoded", "text", sb.String(), "ids", []int32{id})
				continue
			}

			runes := []rune(sb.String())
			merges := make([]merge, len(runes))
			for r := range runes {
				merges[r] = merge{
					p:     r - 1,
					n:     r + 1,
					runes: []rune{runes[r]},
				}
			}

			pairwise := func(a, b int) *pair {
				if a < 0 || b >= len(runes) {
					return nil
				}

				left, right := string(merges[a].runes), string(merges[b].runes)
				rank := bpe.Vocabulary.Merge(left, right)
				if rank < 0 {
					return nil
				}

				return &pair{
					a:     a,
					b:     b,
					rank:  rank,
					value: left + right,
				}
			}

			pairs := heap.NewWith(func(i, j *pair) int {
				return cmp.Compare(i.rank, j.rank)
			})

			for i := range len(runes) - 1 {
				if pair := pairwise(i, i+1); pair != nil {
					pairs.Push(pair)
				}
			}

			for !pairs.Empty() {
				pair, _ := pairs.Pop()

				left, right := merges[pair.a], merges[pair.b]
				if len(left.runes) == 0 || len(right.runes) == 0 ||
					string(left.runes)+string(right.runes) != pair.value {
					continue
				}

				merges[pair.a].runes = append(left.runes, right.runes...)
				merges[pair.b].runes = nil

				merges[pair.a].n = right.n
				if right.n < len(merges) {
					merges[right.n].p = pair.a
				}

				if pair := pairwise(merges[pair.a].p, pair.a); pair != nil {
					pairs.Push(pair)
				}

				if pair := pairwise(pair.a, merges[pair.a].n); pair != nil {
					pairs.Push(pair)
				}
			}

			for _, merge := range merges {
				if len(merge.runes) > 0 {
					// TODO: handle the edge case where the rune isn't in the vocabulary
					if id := bpe.Vocabulary.Encode(string(merge.runes)); id >= 0 {
						ids = append(ids, id)
						slog.Debug("encoded", "text", string(merge.runes), "ids", []int32{id})
					}
				}
			}
		}
	}

	return ids, nil
}

func (bpe BytePairEncoding) Decode(ids []int32) (string, error) {
	var sb strings.Builder
	for _, id := range ids {
		for _, r := range bpe.Vocabulary.Decode(id) {
			switch {
			case r == 0x0100:
				// this produces 0x00 aka NULL
				continue
			case r == 0x0143:
				r = 0x00ad
			case r > 0x0100 && r <= 0x0120:
				r = r - 0x0100
			case r > 0x0120 && r <= 0x0142:
				r = r - 0x00a2
			}

			// NOTE: not using WriteRune here because it writes the UTF-8
			// encoding of the rune which is _not_ what we want
			if err := sb.WriteByte(byte(r)); err != nil {
				return "", err
			}
		}
	}

	slog.Debug("decoded", "ids", ids, "text", sb.String())
	return sb.String(), nil
}
