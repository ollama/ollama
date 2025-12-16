package model

import (
	"cmp"
	"iter"
	"slices"
	"strings"

	"github.com/dlclark/regexp2"
	"github.com/emirpasic/gods/v2/trees/binaryheap"
	"github.com/ollama/ollama/logutil"
)

type BytePairEncoding struct {
	vocab   *Vocabulary
	regexps []*regexp2.Regexp
}

var _ TextProcessor = (*BytePairEncoding)(nil)

func NewBytePairEncoding(vocab *Vocabulary, pretokenizers ...string) BytePairEncoding {
	if len(pretokenizers) == 0 {
		// set default byte-level pretokenizer if none provided, e.g.
		// https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs#L44
		pretokenizers = []string{`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`}
	}

	return BytePairEncoding{
		vocab: vocab,
		regexps: slices.Collect(func(yield func(*regexp2.Regexp) bool) {
			for _, p := range pretokenizers {
				if !yield(regexp2.MustCompile(p, regexp2.RE2)) {
					return
				}
			}
		}),
	}
}

func (bpe BytePairEncoding) Vocabulary() *Vocabulary {
	return bpe.vocab
}

func (bpe BytePairEncoding) Is(id int32, special Special) bool {
	return bpe.vocab.Is(id, special)
}

func (bpe *BytePairEncoding) split(s string) iter.Seq[string] {
	parts := []string{s}
	for _, re := range bpe.regexps {
		parts = slices.Collect(func(yield func(string) bool) {
			for _, part := range parts {
				r := []rune(part)
				var offset int
				for m, _ := re.FindRunesMatch(r); m != nil; m, _ = re.FindNextMatch(m) {
					if offset-m.Index != 0 {
						if !yield(string(r[:m.Index])) {
							return
						}
					}

					if !yield(m.String()) {
						return
					}

					offset = m.Index + m.Length
				}

				if offset < len(r) {
					if !yield(string(r[offset:])) {
						return
					}
				}
			}
		})
	}

	return slices.Values(parts)
}

// fragment is a string fragment and their corresponding token IDs
type fragment struct {
	value string
	ids   []int32
}

// pair is a pair of merges and its rank
type pair[T int | float32] struct {
	a, b *merge
	rank T
}

type merge struct {
	offset, size int
	prev, next   *merge
}

func (bpe BytePairEncoding) Encode(s string, addSpecial bool) ([]int32, error) {
	fragments := []fragment{{value: s}}
	for _, special := range bpe.vocab.SpecialVocabulary() {
		// TODO: process special tokens concurrently
		id := bpe.vocab.Encode(special)
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

		for split := range bpe.split(frag.value) {
			// TODO: process splits concurrently
			var sb strings.Builder
			for _, b := range []byte(split) {
				r := rune(b)
				switch {
				case r == 0x00ad:
					r = 0x0143
				case r <= 0x0020:
					r = r + 0x0100
				case r >= 0x007f && r <= 0x00a0:
					r = r + 0x00a2
				}

				sb.WriteRune(r)
			}

			// short circuit if the fragment is in the vocabulary
			if id := bpe.vocab.Encode(sb.String()); id >= 0 {
				ids = append(ids, id)
				continue
			}

			runes := []rune(sb.String())

			root := &merge{offset: len(runes) - 1, size: 1}
			for i := len(runes) - 2; i >= 0; i-- {
				m := &merge{offset: i, size: 1, next: root}
				root.prev = m
				root = m
			}

			pairwise := func(a, b *merge) *pair[int] {
				if a != nil && b != nil {
					aa := string(runes[a.offset : a.offset+a.size])
					bb := string(runes[b.offset : b.offset+b.size])
					if rank := bpe.vocab.Merge(aa, bb); rank >= 0 {
						return &pair[int]{a: a, b: b, rank: rank}
					}
				}

				return nil
			}

			pairs := binaryheap.NewWith(func(i, j *pair[int]) int { return cmp.Compare(i.rank, j.rank) })
			for m := root; m != nil; m = m.next {
				if pair := pairwise(m, m.next); pair != nil {
					pairs.Push(pair)
				}
			}

			for !pairs.Empty() {
				p, _ := pairs.Pop()
				a := string(runes[p.a.offset : p.a.offset+p.a.size])
				b := string(runes[p.b.offset : p.b.offset+p.b.size])
				if a == "" || b == "" || bpe.vocab.Merge(a, b) != p.rank {
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
				if id := bpe.vocab.Encode(string(runes[m.offset : m.offset+m.size])); id >= 0 {
					ids = append(ids, id)
				}
			}
		}
	}

	if addSpecial {
		ids = bpe.vocab.addSpecials(ids)
	}

	logutil.Trace("encoded", "string", s, "ids", ids)
	return ids, nil
}

func (bpe BytePairEncoding) Decode(ids []int32) (string, error) {
	var sb strings.Builder
	for _, id := range ids {
		for _, r := range bpe.vocab.Decode(id) {
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

	logutil.Trace("decoded", "string", sb.String(), "from", ids)
	return sb.String(), nil
}
