package deepseek_v41

import "fmt"

// engramLayout assigns a disjoint range of table rows to each n-gram/head
// pair. The table dimensions in config.json must exactly match these ranges.
type engramLayout struct {
	layerIDs []int
	primes   [][]int64
	offsets  [][]int64
	ngram    int
	heads    int
}

func newEngramLayout(cfg *Config) (*engramLayout, error) {
	cols := (cfg.EngramMaxNGramSize - 1) * cfg.EngramNumHeads
	if cols <= 0 || len(cfg.EngramLayerIDs) != len(cfg.EngramNumEmbeddings) {
		return nil, fmt.Errorf("invalid engram table layout")
	}
	layout := &engramLayout{
		layerIDs: cfg.EngramLayerIDs,
		primes:   make([][]int64, len(cfg.EngramLayerIDs)),
		offsets:  make([][]int64, len(cfg.EngramLayerIDs)),
		ngram:    cfg.EngramMaxNGramSize,
		heads:    cfg.EngramNumHeads,
	}
	seen := make(map[int64]bool, cols*len(cfg.EngramLayerIDs))
	for layer := range cfg.EngramLayerIDs {
		layout.primes[layer] = make([]int64, cols)
		layout.offsets[layer] = make([]int64, cols)
		var offset int64
		for group := range cfg.EngramMaxNGramSize - 1 {
			candidate := cfg.EngramVocabSize
			for head := range cfg.EngramNumHeads {
				for !isPrime(candidate) || seen[candidate] {
					candidate++
				}
				seen[candidate] = true
				col := group*cfg.EngramNumHeads + head
				layout.primes[layer][col] = candidate
				layout.offsets[layer][col] = offset
				offset += candidate
				candidate++
			}
		}
		if offset != cfg.EngramNumEmbeddings[layer] {
			return nil, fmt.Errorf("engram layer %d has %d hash rows, table declares %d", cfg.EngramLayerIDs[layer], offset, cfg.EngramNumEmbeddings[layer])
		}
	}
	return layout, nil
}

func isPrime(n int64) bool {
	if n < 2 {
		return false
	}
	if n%2 == 0 {
		return n == 2
	}
	for divisor := int64(3); divisor <= n/divisor; divisor += 2 {
		if n%divisor == 0 {
			return false
		}
	}
	return true
}

// engramHasher carries compressed token IDs across successive input chunks.
// A negative history ID marks a sequence boundary or a masked image token.
type engramHasher struct {
	layout      *engramLayout
	tokenMap    []int32
	multipliers [][]int64
	padID       int64
	history     []int64 // newest first
}

func newEngramHasher(layout *engramLayout, tokenMap []int32, multipliers [][]int64, padTokenID int) (*engramHasher, error) {
	if layout == nil || layout.ngram < 2 || padTokenID < 0 || padTokenID >= len(tokenMap) || len(multipliers) != len(layout.layerIDs) {
		return nil, fmt.Errorf("invalid engram hash metadata")
	}
	for layer, values := range multipliers {
		if len(values) != layout.ngram {
			return nil, fmt.Errorf("engram layer %d has %d multipliers, want %d", layer, len(values), layout.ngram)
		}
		for _, value := range values {
			if value <= 0 || value%2 == 0 {
				return nil, fmt.Errorf("invalid engram multiplier %d", value)
			}
		}
	}
	history := make([]int64, layout.ngram-1)
	for i := range history {
		history[i] = -1
	}
	return &engramHasher{layout: layout, tokenMap: tokenMap, multipliers: multipliers, padID: int64(tokenMap[padTokenID]), history: history}, nil
}

// hashChunk returns [token][engram layer][n-gram/head] table row IDs. A nil
// valid slice means all tokens are text. A false entry stops lookback through
// that position, including when the next chunk begins.
func (h *engramHasher) hashChunk(ids []int32, valid []bool) ([][][]int64, error) {
	if valid != nil && len(valid) != len(ids) {
		return nil, fmt.Errorf("engram token mask has %d entries, want %d", len(valid), len(ids))
	}
	for _, id := range ids {
		if id < 0 || int(id) >= len(h.tokenMap) {
			return nil, fmt.Errorf("engram token ID %d outside vocabulary", id)
		}
	}
	rows := make([][][]int64, len(ids))
	for position, id := range ids {
		current := int64(h.tokenMap[id])
		if valid != nil && !valid[position] {
			current = -1
		}
		rows[position] = make([][]int64, len(h.layout.layerIDs))
		for layer := range h.layout.layerIDs {
			rows[position][layer] = make([]int64, (h.layout.ngram-1)*h.layout.heads)
			blocked := false
			var rolling int64
			for shift := range h.layout.ngram {
				token := current
				if shift > 0 {
					token = h.history[shift-1]
				}
				blocked = blocked || token < 0
				if blocked {
					token = h.padID
				}
				rolling ^= token * h.multipliers[layer][shift]
				if shift > 0 {
					for head := range h.layout.heads {
						col := (shift-1)*h.layout.heads + head
						rows[position][layer][col] = h.layout.offsets[layer][col] + rolling%h.layout.primes[layer][col]
					}
				}
			}
		}
		copy(h.history[1:], h.history[:len(h.history)-1])
		h.history[0] = current
	}
	return rows, nil
}
