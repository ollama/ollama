package parser

import (
	"testing"
)

func TestGroupGGUFShards(t *testing.T) {
	tests := []struct {
		name           string
		files          []string
		wantShardSets  int
		wantSingles    int
		firstSetShards int
		firstSetBase   string
	}{
		{
			name: "complete shard set",
			files: []string{
				"model-00001-of-00003.gguf",
				"model-00002-of-00003.gguf",
				"model-00003-of-00003.gguf",
			},
			wantShardSets:  1,
			wantSingles:    0,
			firstSetShards: 3,
			firstSetBase:   "model",
		},
		{
			name: "incomplete shard set",
			files: []string{
				"model-00001-of-00003.gguf",
				"model-00002-of-00003.gguf",
			},
			wantShardSets:  1,
			wantSingles:    0,
			firstSetShards: 2, // Only 2 found
			firstSetBase:   "model",
		},
		{
			name: "mixed sharded and single",
			files: []string{
				"model-00001-of-00002.gguf",
				"model-00002-of-00002.gguf",
				"single-model.gguf",
			},
			wantShardSets:  1,
			wantSingles:    1,
			firstSetShards: 2,
			firstSetBase:   "model",
		},
		{
			name: "only single files",
			files: []string{
				"model1.gguf",
				"model2.gguf",
			},
			wantShardSets: 0,
			wantSingles:   2,
		},
		{
			name: "multiple shard sets",
			files: []string{
				"model-a-00001-of-00002.gguf",
				"model-a-00002-of-00002.gguf",
				"model-b-00001-of-00003.gguf",
				"model-b-00002-of-00003.gguf",
				"model-b-00003-of-00003.gguf",
			},
			wantShardSets:  2,
			wantSingles:    0,
			firstSetShards: 2,
			firstSetBase:   "model-a",
		},
		{
			name:          "empty list",
			files:         []string{},
			wantShardSets: 0,
			wantSingles:   0,
		},
		{
			name: "inconsistent total count",
			files: []string{
				"model-00001-of-00002.gguf",
				"model-00002-of-00003.gguf", // Inconsistent total
			},
			wantShardSets:  1,
			wantSingles:    1, // One creates a shard set, one treated as single
			firstSetShards: 1,
			firstSetBase:   "model",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shardSets, singles := GroupGGUFShards(tt.files)

			if len(shardSets) != tt.wantShardSets {
				t.Errorf("GroupGGUFShards() got %d shard sets, want %d", len(shardSets), tt.wantShardSets)
			}

			if len(singles) != tt.wantSingles {
				t.Errorf("GroupGGUFShards() got %d singles, want %d", len(singles), tt.wantSingles)
			}

			if tt.wantShardSets > 0 {
				firstSet := shardSets[0]
				if len(firstSet.Shards) != tt.firstSetShards {
					t.Errorf("First shard set got %d shards, want %d", len(firstSet.Shards), tt.firstSetShards)
				}
				if firstSet.BaseName != tt.firstSetBase {
					t.Errorf("First shard set basename = %s, want %s", firstSet.BaseName, tt.firstSetBase)
				}
			}
		})
	}
}

func TestValidateShardSet(t *testing.T) {
	tests := []struct {
		name    string
		set     GGUFShardSet
		wantErr bool
	}{
		{
			name: "valid complete set",
			set: GGUFShardSet{
				BaseName:   "model",
				TotalCount: 3,
				Shards: []string{
					"model-00001-of-00003.gguf",
					"model-00002-of-00003.gguf",
					"model-00003-of-00003.gguf",
				},
			},
			wantErr: false,
		},
		{
			name: "incomplete set",
			set: GGUFShardSet{
				BaseName:   "model",
				TotalCount: 3,
				Shards: []string{
					"model-00001-of-00003.gguf",
					"model-00002-of-00003.gguf",
				},
			},
			wantErr: true,
		},
		{
			name: "wrong naming",
			set: GGUFShardSet{
				BaseName:   "model",
				TotalCount: 2,
				Shards: []string{
					"model-00001-of-00002.gguf",
					"wrong-name.gguf",
				},
			},
			wantErr: true,
		},
		{
			name: "empty shards",
			set: GGUFShardSet{
				BaseName:   "model",
				TotalCount: 2,
				Shards:     []string{},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateShardSet(tt.set)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateShardSet() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestShardPattern(t *testing.T) {
	tests := []struct {
		name      string
		filename  string
		wantMatch bool
		wantBase  string
		wantIndex string
		wantTotal string
	}{
		{
			name:      "standard format",
			filename:  "model-00001-of-00003.gguf",
			wantMatch: true,
			wantBase:  "model",
			wantIndex: "00001",
			wantTotal: "00003",
		},
		{
			name:      "complex base name",
			filename:  "ggml-model-f16-00002-of-00004.gguf",
			wantMatch: true,
			wantBase:  "ggml-model-f16",
			wantIndex: "00002",
			wantTotal: "00004",
		},
		{
			name:      "single file",
			filename:  "model.gguf",
			wantMatch: false,
		},
		{
			name:      "wrong extension",
			filename:  "model-00001-of-00003.bin",
			wantMatch: false,
		},
		{
			name:      "wrong format",
			filename:  "model-1-of-3.gguf",
			wantMatch: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			matches := shardPattern.FindStringSubmatch(tt.filename)
			gotMatch := matches != nil

			if gotMatch != tt.wantMatch {
				t.Errorf("shardPattern.FindStringSubmatch() matched = %v, want %v", gotMatch, tt.wantMatch)
				return
			}

			if tt.wantMatch {
				if matches[1] != tt.wantBase {
					t.Errorf("base name = %s, want %s", matches[1], tt.wantBase)
				}
				if matches[2] != tt.wantIndex {
					t.Errorf("index = %s, want %s", matches[2], tt.wantIndex)
				}
				if matches[3] != tt.wantTotal {
					t.Errorf("total = %s, want %s", matches[3], tt.wantTotal)
				}
			}
		})
	}
}
