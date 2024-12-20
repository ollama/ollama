package model

import (
	"reflect"
	"testing"
)

func TestBytePairEncoding(t *testing.T) {
	// Create a simple test vocabulary
	vocab := &Vocabulary{
		Values: []string{
			"Hello",
			"World",
			"!",
			"How",
			"are",
			"you",
			"t",
			"o",
			"d",
			"a",
			"y",
			"to",
			"tod",
			"toda",
			"today",
			" ",
		},
		Types: []uint32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3}, // 3 for special token (space)
		Merges: []string{
			"to",
			"tod",
			"toda",
			"today",
		},
		BOS: 0,
		EOS: 1,
	}

	bpe := BytePairEncoding{
		Pretokenizer: `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		Vocabulary:   vocab,
	}

	tests := []struct {
		name    string
		input   string
		want    []int32
		wantErr bool
	}{
		{
			name:    "simple hello world",
			input:   "Hello World!",
			want:    []int32{0, 15, 1, 2}, // indexes in the vocabulary
			wantErr: false,
		},
		{
			name:    "empty string",
			input:   "",
			want:    []int32{},
			wantErr: false,
		},
		{
			name:    "just spaces",
			input:   "   ",
			want:    []int32{15, 15, 15}, // space token repeated
			wantErr: false,
		},
		{
			name:    "today with merges",
			input:   "today",
			want:    []int32{14}, // should merge
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := bpe.Encode(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("BytePairEncoding.Encode() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("BytePairEncoding.Encode() = %v, want %v", got, tt.want)
			}

			// Test round trip if encoding succeeded
			if err == nil {
				decoded, err := bpe.Decode(got)
				if err != nil {
					t.Errorf("BytePairEncoding.Decode() error = %v", err)
					return
				}
				// Note: The decoded string might not exactly match the input due to
				// tokenization/normalization, so we re-encode it to compare
				reEncoded, err := bpe.Encode(decoded)
				if err != nil {
					t.Errorf("BytePairEncoding.Encode() error on round trip = %v", err)
					return
				}
				if !reflect.DeepEqual(reEncoded, got) {
					t.Errorf("Round trip failed: original tokens = %v, after round trip = %v", got, reEncoded)
				}
			}
		})
	}
}

func TestBytePairEncodingSpecialTokens(t *testing.T) {
	vocab := &Vocabulary{
		Values: []string{
			"<s>",
			"</s>",
			"<pad>",
			"Hello",
			"World",
		},
		Types: []uint32{3, 3, 3, 1, 1}, // 3 for special tokens
		BOS:   0,
		EOS:   1,
	}

	bpe := BytePairEncoding{
		Pretokenizer: `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		Vocabulary:   vocab,
	}

	tests := []struct {
		name    string
		input   string
		want    []int32
		wantErr bool
	}{
		{
			name:    "text with special token at start",
			input:   "<s>Hello",
			want:    []int32{0, 3},
			wantErr: false,
		},
		{
			name:    "text with special token at end",
			input:   "World</s>",
			want:    []int32{4, 1},
			wantErr: false,
		},
		{
			name:    "special token in middle",
			input:   "Hello<pad>World",
			want:    []int32{3, 2, 4},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := bpe.Encode(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("BytePairEncoding.Encode() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("BytePairEncoding.Encode() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBytePairEncodingSplit(t *testing.T) {
	bpe := BytePairEncoding{
		Pretokenizer: `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
	}

	tests := []struct {
		name    string
		input   string
		want    []string
		wantErr bool
	}{
		{
			name:  "basic splitting",
			input: "Hello World!",
			want:  []string{"Hello", " World", "!"},
		},
		{
			name:  "contractions",
			input: "I'm don't won't",
			want:  []string{"I", "'m", " don", "'t", " won", "'t"},
		},
		{
			name:  "numbers",
			input: "In 2024 there are 365 days",
			want:  []string{"In", " ", "202", "4", " there", " are", " ", "365", " days"},
		},
		{
			name:  "special characters",
			input: "Hello!! ...world",
			want:  []string{"Hello", "!!", " ...", "world"},
		},
		{
			name:  "multiple spaces",
			input: "Hello    World",
			want:  []string{"Hello", "   ", " World"},
		},
		{
			name:  "newlines",
			input: "Hello\nWorld",
			want:  []string{"Hello", "\n", "World"},
		},
		{
			name:  "mixed case and punctuation",
			input: "Hello, WORLD!! How's it going?",
			want:  []string{"Hello", ",", " WORLD", "!!", " How", "'s", " it", " going", "?"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := bpe.split(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("BytePairEncoding.split() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("BytePairEncoding.split() = %v, want %v", got, tt.want)
			}
		})
	}
}
