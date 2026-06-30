package convert

import (
	"encoding/json"
	"testing"
)

func TestGemma3nIntermediateSize(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		want    gemma3nIntermediateSize
		wantErr bool
	}{
		{
			name: "scalar",
			json: `8192`,
			want: 8192,
		},
		{
			name: "uniform array",
			json: `[8192,8192,8192]`,
			want: 8192,
		},
		{
			name:    "mixed array",
			json:    `[8192,4096]`,
			wantErr: true,
		},
		{
			name:    "empty array",
			json:    `[]`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got gemma3nIntermediateSize
			err := json.Unmarshal([]byte(tt.json), &got)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("got %d, want %d", got, tt.want)
			}
		})
	}
}
