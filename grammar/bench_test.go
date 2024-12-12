//go:build go1.24

package grammar

import "testing"

func BenchmarkFromSchema(b *testing.B) {
	for tt := range testCases(b) {
		b.Run("", func(b *testing.B) {
			s := []byte(tt.schema)

			b.ReportAllocs()
			for b.Loop() {
				_, err := FromSchema(nil, s)
				if err != nil {
					b.Fatalf("GrammarFromSchema: %v", err)
				}
			}
		})
		return
	}
}
