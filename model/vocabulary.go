package model

import (
	"log/slog"
	"slices"
	"sync"
)

type Special int32

const (
	SpecialBOS Special = iota
	SpecialEOS
)

type Vocabulary struct {
	Values []string
	Types  []int32
	Scores []float32
	Merges []string

	BOS, EOS       []int32
	AddBOS, AddEOS bool

	special []string

	valuesOnce sync.Once
	values     map[string]int32

	mergeOnce sync.Once
	merge     map[string]int32
}

func NewVocabulary(values []string, types []int32, scores []float32, merges []string, bos, eos []int32, addBOS, addEOS bool) *Vocabulary {
	v := &Vocabulary{
		Values: values,
		Types:  types,
		Scores: scores,
		Merges: merges,
		BOS:    bos,
		EOS:    eos,
		AddBOS: addBOS,
		AddEOS: addEOS,
	}
	// Precompute special tokens slice
	v.special = make([]string, 0, len(values)/10)
	for i, t := range v.Types {
		if t == TOKEN_TYPE_CONTROL || t == TOKEN_TYPE_USER_DEFINED {
			v.special = append(v.special, v.Values[i])
		}
	}
	return v
}

func (v *Vocabulary) Is(id int32, special Special) bool {
	switch special {
	case SpecialBOS:
		return slices.Contains(v.BOS, id)
	case SpecialEOS:
		return slices.Contains(v.EOS, id)
	default:
		return false
	}
}

func (v *Vocabulary) addSpecials(ids []int32) []int32 {
	if v.AddBOS && len(v.BOS) > 0 {
		if slices.Contains(v.BOS, ids[0]) {
			slog.Warn("adding bos token to prompt which already has it", "id", v.BOS)
		}

		slog.Debug("adding bos token to prompt", "id", v.BOS)
		ids = append([]int32{v.BOS[0]}, ids...)
	}

	if v.AddEOS && len(v.EOS) > 0 {
		if slices.Contains(v.BOS, ids[len(ids)-1]) {
			slog.Warn("adding eos token to prompt which already has it", "id", v.EOS)
		}

		slog.Debug("adding eos token to prompt", "id", v.EOS)
		ids = append(ids, v.EOS[0])
	}

	return ids
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
