package model

import (
	"container/heap"
	"log/slog"
	"strings"
)

const WhitespaceSeparator = "▁"

type SentencePieceModel struct {
	maxTokenLen int
	vocab       *Vocabulary
}

var _ TextProcessor = (*SentencePieceModel)(nil)

func NewSentencePieceModel(vocab *Vocabulary) SentencePieceModel {
	slog.Debug("Tokens", "num tokens", len(vocab.Values), "vals", vocab.Values[:5], "scores", vocab.Scores[:5], "types", vocab.Types[:5])

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

	slog.Debug("Token counts", "normal", counter[TOKEN_TYPE_NORMAL], "unknown", counter[TOKEN_TYPE_UNKNOWN], "control", counter[TOKEN_TYPE_CONTROL],
		"user defined", counter[TOKEN_TYPE_USER_DEFINED], "unused", counter[TOKEN_TYPE_UNUSED], "byte", counter[TOKEN_TYPE_BYTE],
		"max token len", maxTokenLen)

	return SentencePieceModel{
		maxTokenLen: maxTokenLen,
		vocab:       vocab,
	}
}

func (spm SentencePieceModel) Is(id int32, special Special) bool {
	return spm.vocab.Is(id, special)
}

func (spm SentencePieceModel) Encode(s string, addSpecial bool) ([]int32, error) {
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

		if spm.vocab.Encode(frag.value) >= 0 {
			ids = append(ids, spm.vocab.Encode(frag.value))
			continue
		}

		s := newTokenizer(spm.vocab)
		ids = append(ids, s.tokenize(strings.ReplaceAll(frag.value, " ", WhitespaceSeparator))...)
	}

	if addSpecial && len(ids) > 0 {
		if spm.vocab.AddBOS {
			if ids[0] == spm.vocab.BOS {
				slog.Warn("adding bos token to prompt which already has it", "id", spm.vocab.BOS)
			}

			slog.Debug("adding bos token to prompt", "id", spm.vocab.BOS)
			ids = append([]int32{spm.vocab.BOS}, ids...)
		}

		if spm.vocab.AddEOS {
			if ids[len(ids)-1] == spm.vocab.EOS {
				slog.Warn("adding eos token to prompt which already has it", "id", spm.vocab.EOS)
			}

			slog.Debug("adding eos token to prompt", "id", spm.vocab.EOS)
			ids = append(ids, spm.vocab.EOS)
		}
	}

	return ids, nil
}

func (spm SentencePieceModel) Decode(ids []int32) (string, error) {
	var sb strings.Builder
	for _, id := range ids {
		data := spm.vocab.Decode(id)
		data = strings.ReplaceAll(data, WhitespaceSeparator, " ")
		if _, err := sb.WriteString(data); err != nil {
			return "", err
		}
	}

	return sb.String(), nil
}

type symbol struct {
	text string
	prev int
	next int
}

type tokenizer struct {
	heap    *bigramHeap
	history map[string][2]int
	symbols []symbol
	vocab   *Vocabulary
}

func newTokenizer(vocab *Vocabulary) *tokenizer {
	h := &bigramHeap{}
	heap.Init(h)

	return &tokenizer{
		heap:    h,
		history: make(map[string][2]int),
		symbols: []symbol{},
		vocab:   vocab,
	}
}

func (t *tokenizer) tokenize(text string) []int32 {
	var symbols []symbol
	for i, r := range []rune(text) {
		sym := symbol{
			text: string(r),
			prev: i - 1,
			next: i + 1,
		}
		symbols = append(symbols, sym)
	}

	if len(symbols) == 0 {
		return []int32{}
	}

	symbols[len(symbols)-1].next = -1

	// Add initial bigrams to the queue
	for i := range len(symbols) - 1 {
		t.add(i, i+1, symbols)
	}

	// Process bigrams in order of score
	for t.heap.Len() > 0 {
		bigram := heap.Pop(t.heap).(*bigram)
		left := &symbols[bigram.left]
		right := &symbols[bigram.right]

		if left.text == "" || right.text == "" || len(left.text)+len(right.text) != bigram.size {
			continue
		}

		left.text += right.text
		right.text = ""

		left.next = right.next
		if right.next != -1 {
			symbols[right.next].prev = bigram.left
		}

		t.add(left.prev, bigram.left, symbols)
		t.add(bigram.left, left.next, symbols)
	}

	var output []int32
	for i := 0; i != -1; i = symbols[i].next {
		if symbols[i].text != "" {
			tokens := t.resegment(symbols[i].text, symbols)
			output = append(output, tokens...)
		}
	}

	return output
}

// resegment recursively breaks down tokens that couldn't be encoded
func (t *tokenizer) resegment(text string, symbols []symbol) []int32 {
	id := t.vocab.Encode(text)

	if id >= 0 {
		return []int32{id}
	}

	if pair, exists := t.history[text]; exists {
		left := t.resegment(symbols[pair[0]].text, symbols)
		right := t.resegment(symbols[pair[1]].text, symbols)
		return append(left, right...)
	}

	var result []int32
	for _, b := range []byte(text) {
		unknownID := t.vocab.Encode(string(b))
		if unknownID >= 0 {
			result = append(result, unknownID)
		} else {
			slog.Debug("unknown byte token", "byte", b)
		}
	}

	return result
}

// add creates a new bigram and adds it to the heap
func (t *tokenizer) add(left, right int, symbols []symbol) {
	if left == -1 || right == -1 {
		return
	}

	leftSym := &symbols[left]
	rightSym := &symbols[right]

	if leftSym.text == "" || rightSym.text == "" {
		return
	}

	combined := leftSym.text + rightSym.text
	id := t.vocab.Encode(combined)

	if id < 0 || int(id) >= len(t.vocab.Scores) {
		return
	}

	bigram := &bigram{
		left:  left,
		right: right,
		score: t.vocab.Scores[id],
		size:  len(combined),
	}

	heap.Push(t.heap, bigram)
	t.history[combined] = [2]int{left, right}
}

// bigram represents a pair of adjacent symbols
type bigram struct {
	left  int
	right int
	score float32
	size  int
}

type bigramHeap []*bigram

func (bq bigramHeap) Len() int { return len(bq) }

func (bq bigramHeap) Less(i, j int) bool {
	return (bq[i].score > bq[j].score) || (bq[i].score == bq[j].score && bq[i].left < bq[j].left)
}

func (bq bigramHeap) Swap(i, j int) { bq[i], bq[j] = bq[j], bq[i] }

func (bq *bigramHeap) Push(x interface{}) {
	item := x.(*bigram)
	*bq = append(*bq, item)
}

func (bq *bigramHeap) Pop() interface{} {
	old := *bq
	n := len(old)
	item := old[n-1]
	*bq = old[0 : n-1]
	return item
}
