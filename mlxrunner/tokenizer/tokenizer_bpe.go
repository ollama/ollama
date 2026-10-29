package tokenizer

import "unicode/utf8"

type bpeMergeNode struct {
	start int
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

type bpePairHeap []bpePair

func (h bpePairHeap) less(i, j int) bool {
	return h[i].rank < h[j].rank || (h[i].rank == h[j].rank && h[i].left < h[j].left)
}

// Store pairs by value so each candidate merge does not allocate separately.
func (h bpePairHeap) push(pair bpePair) bpePairHeap {
	h = append(h, pair)
	for i := len(h) - 1; i > 0; {
		parent := (i - 1) / 2
		if !h.less(i, parent) {
			break
		}
		h[i], h[parent] = h[parent], h[i]
		i = parent
	}
	return h
}

func (h bpePairHeap) pop() (bpePairHeap, bpePair) {
	pair := h[0]
	last := len(h) - 1
	h[0] = h[last]
	h[last] = bpePair{}
	h = h[:last]
	for i := 0; 2*i+1 < len(h); {
		child := 2*i + 1
		if child+1 < len(h) && h.less(child+1, child) {
			child++
		}
		if !h.less(child, i) {
			break
		}
		h[i], h[child] = h[child], h[i]
		i = child
	}
	return h, pair
}

// encodeBPEMerge merges the lowest-rank valid pair, breaking ties left to right.
// Only neighboring pairs need to be rechecked after each merge.
func (t *Tokenizer) encodeBPEMerge(encoded string, ids []int32) []int32 {
	if encoded == "" {
		return ids
	}

	// Normalize malformed UTF-8 to replacement runes before using byte offsets.
	if !utf8.ValidString(encoded) {
		encoded = string([]rune(encoded))
	}
	// Most pretokenized pieces fit here; append grows for longer pieces.
	nodes := make([]bpeMergeNode, 0, 32)
	for offset, r := range encoded {
		i := len(nodes)
		nodes = append(nodes, bpeMergeNode{
			start: offset,
			prev:  i - 1,
			next:  i + 1,
			token: encoded[offset : offset+utf8.RuneLen(r)],
		})
	}

	pairwise := func(left, right int) (bpePair, bool) {
		if left < 0 || right >= len(nodes) {
			return bpePair{}, false
		}
		if nodes[left].token == "" || nodes[right].token == "" {
			return bpePair{}, false
		}

		leftToken, rightToken := nodes[left].token, nodes[right].token
		rank, ok := t.vocab.Merges[leftToken+" "+rightToken]
		if !ok {
			return bpePair{}, false
		}

		// Merged tokens remain contiguous spans of the encoded input.
		value := encoded[nodes[left].start : nodes[right].start+len(rightToken)]
		if _, ok := t.vocab.Reverse[value]; !ok {
			return bpePair{}, false
		}

		return bpePair{
			left:  left,
			right: right,
			rank:  rank,
			value: value,
		}, true
	}

	pairs := make(bpePairHeap, 0, 32)
	for i := range len(nodes) - 1 {
		if pair, ok := pairwise(i, i+1); ok {
			pairs = pairs.push(pair)
		}
	}

	for len(pairs) > 0 {
		var pair bpePair
		pairs, pair = pairs.pop()
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

		if pair, ok := pairwise(nodes[pair.left].prev, pair.left); ok {
			pairs = pairs.push(pair)
		}
		if pair, ok := pairwise(pair.left, nodes[pair.left].next); ok {
			pairs = pairs.push(pair)
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
