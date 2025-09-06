package model

import (
	"fmt"
	"testing"
)

// genValues builds a Vocabulary with N tokens, marking every 10th as CONTROL.
func genValues(n int) ([]string, []int32) {
	vals := make([]string, n)
	tys := make([]int32, n)
	for i := 0; i < n; i++ {
		vals[i] = fmt.Sprintf("tok%d", i)
		if i%10 == 0 {
			tys[i] = TOKEN_TYPE_CONTROL
		} else {
			tys[i] = TOKEN_TYPE_USER_DEFINED
		}
	}
	return vals, tys
}

func BenchmarkSpecialVocabulary(b *testing.B) {
	vals, tys := genValues(10000)
	// Initialize vocab ONCE outside measured region.
	v := NewVocabulary(vals, tys, nil, nil, nil, nil, false, false)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = v.SpecialVocabulary()
	}
}
