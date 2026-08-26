package model

import (
	"strings"
	"testing"
)

func TestConfigVocabSize(t *testing.T) {
	tests := []struct {
		name    string
		config  string
		want    int
		wantErr string
	}{
		{name: "top level", config: `{"vocab_size": 128}`, want: 128},
		{name: "nested text config", config: `{"vocab_size": 1, "text_config": {"vocab_size": 256}}`, want: 256},
		{name: "null text config", config: `{"vocab_size": 128, "text_config": null}`, want: 128},
		{name: "missing", config: `{}`, wantErr: "missing vocab_size"},
		{name: "invalid text config", config: `{"text_config": 1}`, wantErr: "parse text_config vocab_size"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := configVocabSize([]byte(tt.config))
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("vocab size = %d, want %d", got, tt.want)
			}
		})
	}
}
