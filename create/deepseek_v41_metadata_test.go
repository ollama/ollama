package create

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDeepSeekV41EngramImportRejectsMalformedSource(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"wrong":"tokenizer"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{}`), 0o600); err != nil {
		t.Fatal(err)
	}
	called := false
	_, err := importDeepSeekV41EngramMetadata(context.Background(), dir, blobStoreFunc(func(_ io.Reader, _, _ string) (LayerInfo, error) {
		called = true
		return LayerInfo{}, nil
	}))
	if err == nil || !strings.Contains(err.Error(), "unsupported DeepSeek V4.1 engram hash layout") || called {
		t.Fatalf("malformed source: error %v, blob write called %t", err, called)
	}
}

func TestNormalizeDeepSeekV41Token(t *testing.T) {
	for _, tc := range []struct{ input, want string }{
		{" Café\t", "cafe"},
		{"A  B\n", "a b"},
		{" ", " "},
	} {
		if got := normalizeDeepSeekV41Token(tc.input); got != tc.want {
			t.Errorf("normalizeDeepSeekV41Token(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}
