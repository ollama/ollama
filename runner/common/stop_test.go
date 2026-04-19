package common

import (
	"reflect"
	"strings"
	"testing"
)

func TestFindStopAfterAppend(t *testing.T) {
	tests := []struct {
		name  string
		piece []string
		stops []string
		want  bool
		stop  string
	}{
		{
			name:  "within last piece",
			piece: []string{"hello ", "stop and more"},
			stops: []string{"stop"},
			want:  true,
			stop:  "stop",
		},
		{
			name:  "spans previous suffix",
			piece: []string{"hello st", "op"},
			stops: []string{"stop"},
			want:  true,
			stop:  "stop",
		},
		{
			name:  "spans multiple previous pieces",
			piece: []string{"hello ", "s", "t", "op"},
			stops: []string{"stop"},
			want:  true,
			stop:  "stop",
		},
		{
			name:  "already checked prefix is ignored",
			piece: []string{"stop", " and more"},
			stops: []string{"stop"},
			want:  false,
		},
		{
			name:  "empty stop",
			piece: []string{"anything"},
			stops: []string{""},
			want:  true,
		},
		{
			name:  "preserves stop list priority across boundary",
			piece: []string{"ab", "cd zz"},
			stops: []string{"abcd", "zz"},
			want:  true,
			stop:  "abcd",
		},
		{
			name:  "ignores previous only shorter stop",
			piece: []string{"abc", "d"},
			stops: []string{"bc", "unmatched-long-stop"},
			want:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, stop := FindStopAfterAppend(tt.piece, tt.stops)
			if got != tt.want || stop != tt.stop {
				t.Fatalf("FindStopAfterAppend(%q, %q) = %v, %q; want %v, %q", tt.piece, tt.stops, got, stop, tt.want, tt.stop)
			}
		})
	}
}

func TestPieceStopChecksMatchJoinedSuffixChecks(t *testing.T) {
	tests := []struct {
		name   string
		pieces []string
		stops  []string
	}{
		{
			name:   "simple suffix",
			pieces: []string{"hello ", "st"},
			stops:  []string{"stop"},
		},
		{
			name:   "full stop across pieces",
			pieces: []string{"hello ", "st", "op"},
			stops:  []string{"stop"},
		},
		{
			name:   "stop inside newest piece",
			pieces: []string{"hello ", "stop and more"},
			stops:  []string{"stop"},
		},
		{
			name:   "unicode suffix",
			pieces: []string{"hello", string([]byte{0xe0, 0xa0})},
			stops:  []string{"world"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for i := range tt.pieces {
				pieces := tt.pieces[:i+1]
				joined := strings.Join(pieces, "")

				if got, want := ContainsStopSuffixInPieces(pieces, tt.stops), ContainsStopSuffix(joined, tt.stops); got != want {
					t.Fatalf("ContainsStopSuffixInPieces(%q, %q) = %v; want %v", pieces, tt.stops, got, want)
				}
				if got, want := IncompleteUnicodeInPieces(pieces), IncompleteUnicode(joined); got != want {
					t.Fatalf("IncompleteUnicodeInPieces(%q) = %v; want %v", pieces, got, want)
				}

				previousStop := false
				if i > 0 {
					previousStop, _ = FindStop(strings.Join(tt.pieces[:i], ""), tt.stops)
				}
				if previousStop {
					continue
				}

				got, gotStop := FindStopAfterAppend(pieces, tt.stops)
				want, wantStop := FindStop(joined, tt.stops)
				if got != want || gotStop != wantStop {
					t.Fatalf("FindStopAfterAppend(%q, %q) = %v, %q; want %v, %q", pieces, tt.stops, got, gotStop, want, wantStop)
				}
			}
		})
	}
}

func TestTruncateStop(t *testing.T) {
	tests := []struct {
		name          string
		pieces        []string
		stop          string
		expected      []string
		expectedTrunc bool
	}{
		{
			name:          "Single word",
			pieces:        []string{"hello", "world"},
			stop:          "world",
			expected:      []string{"hello"},
			expectedTrunc: false,
		},
		{
			name:          "Partial",
			pieces:        []string{"hello", "wor"},
			stop:          "or",
			expected:      []string{"hello", "w"},
			expectedTrunc: true,
		},
		{
			name:          "Suffix",
			pieces:        []string{"Hello", " there", "!"},
			stop:          "!",
			expected:      []string{"Hello", " there"},
			expectedTrunc: false,
		},
		{
			name:          "Suffix partial",
			pieces:        []string{"Hello", " the", "re!"},
			stop:          "there!",
			expected:      []string{"Hello", " "},
			expectedTrunc: true,
		},
		{
			name:          "Middle",
			pieces:        []string{"hello", " wor"},
			stop:          "llo w",
			expected:      []string{"he"},
			expectedTrunc: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, resultTrunc := TruncateStop(tt.pieces, tt.stop)
			if !reflect.DeepEqual(result, tt.expected) || resultTrunc != tt.expectedTrunc {
				t.Errorf("truncateStop(%v, %s): have %v (%v); want %v (%v)", tt.pieces, tt.stop, result, resultTrunc, tt.expected, tt.expectedTrunc)
			}
		})
	}
}

func BenchmarkPendingStopChecks(b *testing.B) {
	pieces := make([]string, 256)
	for i := range pieces {
		pieces[i] = "partial "
	}
	pieces[len(pieces)-1] = "sto"
	stops := []string{"stop sequence", "another stop"}

	b.Run("join", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			joined := strings.Join(pieces, "")
			FindStop(joined, stops)
			ContainsStopSuffix(joined, stops)
			IncompleteUnicode(joined)
		}
	})

	b.Run("pieces", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			FindStopAfterAppend(pieces, stops)
			ContainsStopSuffixInPieces(pieces, stops)
			IncompleteUnicodeInPieces(pieces)
		}
	})
}

func TestIncompleteUnicode(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected bool
	}{
		{
			name:     "Basic",
			input:    "hi",
			expected: false,
		},
		{
			name:     "Two byte",
			input:    "hi" + string([]byte{0xc2, 0xa3}),
			expected: false,
		},
		{
			name:     "Two byte - missing last",
			input:    "hi" + string([]byte{0xc2}),
			expected: true,
		},
		{
			name:     "Three byte",
			input:    "hi" + string([]byte{0xe0, 0xA0, 0x80}),
			expected: false,
		},
		{
			name:     "Three byte - missing last",
			input:    "hi" + string([]byte{0xe0, 0xA0}),
			expected: true,
		},
		{
			name:     "Three byte - missing last 2",
			input:    "hi" + string([]byte{0xe0}),
			expected: true,
		},
		{
			name:     "Four byte",
			input:    "hi" + string([]byte{0xf0, 0x92, 0x8a, 0xb7}),
			expected: false,
		},
		{
			name:     "Four byte - missing last",
			input:    "hi" + string([]byte{0xf0, 0x92, 0x8a}),
			expected: true,
		},
		{
			name:     "Four byte - missing last 2",
			input:    "hi" + string([]byte{0xf0, 0x92}),
			expected: true,
		},
		{
			name:     "Four byte - missing last 3",
			input:    "hi" + string([]byte{0xf0}),
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IncompleteUnicode(tt.input)
			if result != tt.expected {
				t.Errorf("incompleteUnicode(%s): have %v; want %v", tt.input, result, tt.expected)
			}
		})
	}
}
