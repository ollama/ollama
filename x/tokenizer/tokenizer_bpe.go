//go:build mlx

package tokenizer

import "container/heap"

type bpeMergeNode struct {
	prev  int
	next  int
	token string
}

type bpePair struct {
	left  int
	right int
	rank  int
	value string
}

type bpePairHeap []*bpePair

func (h bpePairHeap) Len() int { return len(h) }

func (h bpePairHeap) Less(i, j int) bool {
	return h[i].rank < h[j].rank || (h[i].rank == h[j].rank && h[i].left < h[j].left)
}

func (h bpePairHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *bpePairHeap) Push(x any) {
	*h = append(*h, x.(*bpePair))
}

func (h *bpePairHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// encodeBPEMerge encodes using BPE merge algorithm.
// Uses the heap/linked-list pair merge strategy from tokenizer/bytepairencoding.go:
// merge the lowest-rank valid pair, then only recheck adjacent pairs.
func (t *Tokenizer) encodeBPEMerge(encoded string, ids []int32) []int32 {
	runes := []rune(encoded)
	if len(runes) == 0 {
		return ids
	}

	nodes := make([]bpeMergeNode, len(runes))
	for i := range runes {
		nodes[i] = bpeMergeNode{
			prev:  i - 1,
			next:  i + 1,
			token: string(runes[i]),
		}
	}

	pairwise := func(left, right int) *bpePair {
		if left < 0 || right >= len(nodes) {
			return nil
		}
		if nodes[left].token == "" || nodes[right].token == "" {
			return nil
		}

		leftToken, rightToken := nodes[left].token, nodes[right].token
		rank, ok := t.vocab.Merges[leftToken+" "+rightToken]
		if !ok {
			return nil
		}

		value := leftToken + rightToken
		if _, ok := t.vocab.Reverse[value]; !ok {
			return nil
		}

		return &bpePair{
			left:  left,
			right: right,
			rank:  rank,
			value: value,
		}
	}

	pairs := bpePairHeap{}
	heap.Init(&pairs)
	for i := 0; i < len(runes)-1; i++ {
		if pair := pairwise(i, i+1); pair != nil {
			heap.Push(&pairs, pair)
		}
	}

	for pairs.Len() > 0 {
		pair := heap.Pop(&pairs).(*bpePair)
		left, right := nodes[pair.left], nodes[pair.right]
		if left.token == "" || right.token == "" {
			continue
		}
		if left.next != pair.right || right.prev != pair.left {
			continue
		}
		if left.token+right.token != pair.value {
			continue
		}

		nodes[pair.left].token = pair.value
		nodes[pair.right].token = ""
		nodes[pair.left].next = right.next
		if right.next < len(nodes) {
			nodes[right.next].prev = pair.left
		}

		if pair := pairwise(nodes[pair.left].prev, pair.left); pair != nil {
			heap.Push(&pairs, pair)
		}
		if pair := pairwise(pair.left, nodes[pair.left].next); pair != nil {
			heap.Push(&pairs, pair)
		}
	}

	for _, node := range nodes {
		if node.token == "" {
			continue
		}

		if id, ok := t.vocab.Reverse[node.token]; ok {
			ids = append(ids, id)
			continue
		}

		ids = t.appendByteFallback(ids, node.token)
	}

	return ids
}

func (t *Tokenizer) appendByteFallback(ids []int32, token string) []int32 {
	if t.typ == TokenizerBPE {
		for _, r := range token {
			if b, ok := decodeByteLevelRune(r); ok {
				if id := t.vocab.byteTokens[b]; id >= 0 {
					ids = append(ids, id)
				}
			}
		}
		return ids
	}

	// SentencePiece fallback uses the UTF-8 bytes for <0xNN> tokens.
	for _, b := range []byte(token) {
		if id := t.vocab.byteTokens[b]; id >= 0 {
			ids = append(ids, id)
		}
	}
	return ids
}

func decodeByteLevelRune(r rune) (byte, bool) {
	switch {
	case r >= 0x00 && r <= 0xFF:
		return byte(r), true
	case r == 0x0100:
		return 0x00, true
	case r == 0x0143:
		return 0x00ad, true
	case r > 0x0100 && r <= 0x0120:
		return byte(r - 0x0100), true
	case r > 0x0120 && r <= 0x0142:
		return byte(r - 0x00a2), true
	default:
		return 0, false
	}
}
