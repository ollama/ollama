package kvcache

import "testing"

func BenchmarkWorking(b *testing.B) {
	b.Run("Paged", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = i * 2
		}
	})
}
