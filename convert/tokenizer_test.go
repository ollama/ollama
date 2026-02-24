package convert

import (
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func createTokenizerFS(t *testing.T, dir string, files map[string]io.Reader) fs.FS {
	t.Helper()

	for k, v := range files {
		if err := func() error {
			f, err := os.Create(filepath.Join(dir, k))
			if err != nil {
				return err
			}
			defer f.Close()

			if _, err := io.Copy(f, v); err != nil {
				return err
			}

			return nil
		}(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	return os.DirFS(dir)
}

func TestParseTokenizer(t *testing.T) {
	cases := []struct {
		name              string
		fsys              fs.FS
		specialTokenTypes []string
		want              *Tokenizer
	}{
		{
			name: "string chat template",
			fsys: createTokenizerFS(t, t.TempDir(), map[string]io.Reader{
				"tokenizer.json": strings.NewReader(`{}`),
				"tokenizer_config.json": strings.NewReader(`{
					"chat_template": "<default template>"
				}`),
			}),
			want: &Tokenizer{
				Vocabulary: &Vocabulary{Model: "gpt2"},
				Pre:        "default",
				Template:   "<default template>",
			},
		},
		{
			name: "list chat template",
			fsys: createTokenizerFS(t, t.TempDir(), map[string]io.Reader{
				"tokenizer.json": strings.NewReader(`{}`),
				"tokenizer_config.json": strings.NewReader(`{
					"chat_template": [
						{
							"name": "default",
							"template": "<default template>"
						},
						{
							"name": "tools",
							"template": "<tools template>"
						}
					]
				}`),
			}),
			want: &Tokenizer{
				Vocabulary: &Vocabulary{Model: "gpt2"},
				Pre:        "default",
				Template:   "<default template>",
			},
		},
		{
			name: "added tokens",
			fsys: createTokenizerFS(t, t.TempDir(), map[string]io.Reader{
				"tokenizer.json": strings.NewReader(`{
					"added_tokens": [
						{
							"id": 999,
							"content": "<unused999>",
							"special": false
						}
					]
				}`),
			}),
			want: &Tokenizer{
				Vocabulary: &Vocabulary{
					Model:  "gpt2",
					Tokens: []string{"<unused999>"},
					Scores: []float32{999},
					Types:  []int32{4},
				},
				Pre: "default",
			},
		},
		{
			name: "added tokens overlap vocab",
			fsys: createTokenizerFS(t, t.TempDir(), map[string]io.Reader{
				"tokenizer.json": strings.NewReader(`{
					"added_tokens": [
						{
							"id": 0,
							"content": "<pad>",
							"special": true
						}
					],
					"model": {
						"vocab": {
							"<pad>": 0
						}
					}
				}`),
			}),
			want: &Tokenizer{
				Vocabulary: &Vocabulary{
					Model:  "gpt2",
					Tokens: []string{"<pad>"},
					Scores: []float32{0},
					Types:  []int32{3},
				},
				Pre: "default",
			},
		},
		{
			name: "special token types",
			fsys: createTokenizerFS(t, t.TempDir(), map[string]io.Reader{
				"tokenizer.json": strings.NewReader(`{
					"added_tokens": [
						{
							"id": 0,
							"content": "<pad>",
							"special": true
						},
						{
							"id": 1,
							"content": "<eos>",
							"special": true
						},
						{
							"id": 2,
							"content": "<bos>",
							"special": true
						},
						{
							"id": 3,
							"content": "<unk>",
							"special": true
						}
					],
					"model": {
						"vocab": {
							"<pad>": 0,
							"<eos>": 1,
							"<bos>": 2,
							"<unk>": 3
						}
					}
				}`),
				"tokenizer_config.json": strings.NewReader(`{
					"add_bos_token": true,
					"add_eos_token": false,
					"bos_token": "<bos>",
					"eos_token": "<eos>",
					"pad_token": "<pad>",
					"unk_token": "<unk>"
				}`),
			}),
			specialTokenTypes: []string{"pad", "eos", "bos", "unk"},
			want: &Tokenizer{
				Vocabulary: &Vocabulary{
					Model:  "gpt2",
					Tokens: []string{"<pad>", "<eos>", "<bos>", "<unk>"},
					Scores: []float32{0, 1, 2, 3},
					Types:  []int32{3, 3, 3, 3},
				},
				SpecialVocabulary: []*SpecialVocabulary{
					{Type: "pad", Content: "<pad>", ID: 0, AddToken: false},
					{Type: "eos", Content: "<eos>", ID: 1, AddToken: false},
					{Type: "bos", Content: "<bos>", ID: 2, AddToken: true},
					{Type: "unk", Content: "<unk>", ID: 3, AddToken: false},
				},
				Pre: "default",
			},
		},
		{
			name: "llama-bpe pretokenizer and control tokens",
			fsys: createTokenizerFS(t, t.TempDir(), map[string]io.Reader{
				"tokenizer.json": strings.NewReader(`{
					"added_tokens": [
						{"id": 1, "content": "<|startoftext|>", "special": true},
						{"id": 6, "content": "<|im_start|>", "special": true},
						{"id": 7, "content": "<|im_end|>", "special": true},
						{"id": 8, "content": "<|tool_list_start|>", "special": true},
						{"id": 9, "content": "<|tool_list_end|>", "special": true},
						{"id": 10, "content": "<|tool_call_start|>", "special": true},
						{"id": 11, "content": "<|tool_call_end|>", "special": true},
						{"id": 12, "content": "<|tool_response_start|>", "special": true},
						{"id": 13, "content": "<|tool_response_end|>", "special": true},
						{"id": 396, "content": "<image>", "special": true},
						{"id": 64400, "content": "<think>", "special": true},
						{"id": 64401, "content": "</think>", "special": true}
					],
					"model": {
						"vocab": {
							"<|startoftext|>": 1,
							"<|im_start|>": 6,
							"<|im_end|>": 7,
							"<|tool_list_start|>": 8,
							"<|tool_list_end|>": 9,
							"<|tool_call_start|>": 10,
							"<|tool_call_end|>": 11,
							"<|tool_response_start|>": 12,
							"<|tool_response_end|>": 13,
							"<image>": 396,
							"<think>": 64400,
							"</think>": 64401
						}
					},
					"pre_tokenizer": {
						"type": "Sequence",
						"pretokenizers": [
							{
								"type": "Split",
								"pattern": {
									"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
								},
								"behavior": "Isolated",
								"invert": false
							},
							{
								"type": "ByteLevel",
								"add_prefix_space": false,
								"trim_offsets": true,
								"use_regex": false
							}
						]
					}
				}`),
			}),
			want: &Tokenizer{
				Vocabulary: &Vocabulary{
					Model: "gpt2",
					Tokens: []string{
						"<|startoftext|>",
						"<|im_start|>",
						"<|im_end|>",
						"<|tool_list_start|>",
						"<|tool_list_end|>",
						"<|tool_call_start|>",
						"<|tool_call_end|>",
						"<|tool_response_start|>",
						"<|tool_response_end|>",
						"<image>",
						"<think>",
						"</think>",
					},
					Scores: []float32{1, 6, 7, 8, 9, 10, 11, 12, 13, 396, 64400, 64401},
					Types:  []int32{3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
				},
				Pre: "llama-bpe",
			},
		},
		{
			name: "list string merges",
			fsys: createTokenizerFS(t, t.TempDir(), map[string]io.Reader{
				"tokenizer.json": strings.NewReader(`{
					"model": {
						"merges": [
							"a b",
							"c d",
							"e f"
						]
					}
				}`),
			}),
			want: &Tokenizer{
				Vocabulary: &Vocabulary{
					Model: "gpt2",
				},
				Merges: []string{
					"a b",
					"c d",
					"e f",
				},
				Pre: "default",
			},
		},
		{
			name: "list list string merges",
			fsys: createTokenizerFS(t, t.TempDir(), map[string]io.Reader{
				"tokenizer.json": strings.NewReader(`{
					"model": {
						"merges": [
							[
								"a", "b"
							],
							[
								"c", "d"
							],
							[
								"e", "f"
							]
						]
					}
				}`),
			}),
			want: &Tokenizer{
				Vocabulary: &Vocabulary{
					Model: "gpt2",
				},
				Merges: []string{
					"a b",
					"c d",
					"e f",
				},
				Pre: "default",
			},
		},
		{
			name: "generation config eos token ids",
			fsys: createTokenizerFS(t, t.TempDir(), map[string]io.Reader{
				"tokenizer.json": strings.NewReader(`{
					"added_tokens": [
						{
							"id": 0,
							"content": "<bos>",
							"special": true
						},
						{
							"id": 1,
							"content": "<eos>",
							"special": true
						},
						{
							"id": 2,
							"content": "<eot>",
							"special": true
						},
						{
							"id": 3,
							"content": "<eom>",
							"special": true
						}
					],
					"model": {
						"vocab": {
							"<bos>": 0,
							"<eos>": 1,
							"<eot>": 2,
							"<eom>": 3
						}
					}
				}`),
				"tokenizer_config.json": strings.NewReader(`{
					"add_bos_token": true,
					"add_eos_token": false,
					"bos_token": "<bos>",
					"eos_token": "<eos>"
				}`),
				"generation_config.json": strings.NewReader(`{
					"bos_token_id": 0,
					"eos_token_id": [1, 2, 3]
				}`),
			}),
			specialTokenTypes: []string{"pad", "eos", "bos", "unk"},
			want: &Tokenizer{
				Vocabulary: &Vocabulary{
					Model:  "gpt2",
					Tokens: []string{"<bos>", "<eos>", "<eot>", "<eom>"},
					Scores: []float32{0, 1, 2, 3},
					Types:  []int32{3, 3, 3, 3},
				},
				SpecialVocabulary: []*SpecialVocabulary{
					{Type: "eos", Content: "<eos>", ID: 1, IDs: []int32{1, 2, 3}, AddToken: false},
					{Type: "bos", Content: "<bos>", ID: 0, AddToken: true},
				},
				Pre: "default",
			},
		},
		{
			name: "qwen35 pretokenizer",
			fsys: createTokenizerFS(t, t.TempDir(), map[string]io.Reader{
				"tokenizer.json": strings.NewReader(`{
					"pre_tokenizer": {
						"type": "Sequence",
						"pretokenizers": [
							{
								"type": "Split",
								"pattern": {
									"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
								}
							}
						]
					}
				}`),
			}),
			want: &Tokenizer{
				Vocabulary: &Vocabulary{Model: "gpt2"},
				Pre:        "qwen35",
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			tokenizer, err := parseTokenizer(tt.fsys, tt.specialTokenTypes)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if diff := cmp.Diff(tt.want, tokenizer); diff != "" {
				t.Errorf("unexpected tokenizer (-want +got):\n%s", diff)
			}
		})
	}
}
