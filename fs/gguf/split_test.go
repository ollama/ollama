package gguf

import "testing"

func TestSplitShardName(t *testing.T) {
	cases := []struct {
		name   string
		prefix string
		index  int
		count  int
		ok     bool
	}{
		{"model-00001-of-00002.gguf", "model", 0, 2, true},
		{"model-00002-of-00002.gguf", "model", 1, 2, true},
		{"Qwen3-Coder-Next-Q4_1-00012-of-00042.gguf", "Qwen3-Coder-Next-Q4_1", 11, 42, true},
		{"/tmp/models/model-00003-of-00003.gguf", "model", 2, 3, true},
		{"dir/model-00001-of-00001.gguf", "model", 0, 1, true},
		{"model.gguf", "", 0, 0, false},
		{"model-00000-of-00002.gguf", "", 0, 0, false},
		{"model-00001-of-00000.gguf", "", 0, 0, false},
		{"model-1-of-2.gguf", "", 0, 0, false},
		{"model-000001-of-000002.gguf", "", 0, 0, false},
		{"model-00001-of-00002.bin", "", 0, 0, false},
		{"model-00001-of-00002", "", 0, 0, false},
		{"", "", 0, 0, false},
	}

	for _, c := range cases {
		prefix, index, count, ok := SplitShardName(c.name)
		if ok != c.ok || prefix != c.prefix || index != c.index || count != c.count {
			t.Errorf("SplitShardName(%q) = (%q, %d, %d, %v), want (%q, %d, %d, %v)",
				c.name, prefix, index, count, ok, c.prefix, c.index, c.count, c.ok)
		}
	}
}

func TestSplitShardFileName(t *testing.T) {
	cases := []struct {
		prefix, want string
		index, count int
	}{
		{"model", "model-00001-of-00002.gguf", 0, 2},
		{"model", "model-00002-of-00002.gguf", 1, 2},
		{"Qwen3-Coder-Next-Q4_1", "Qwen3-Coder-Next-Q4_1-00012-of-00042.gguf", 11, 42},
	}

	for _, c := range cases {
		if got := SplitShardFileName(c.prefix, c.index, c.count); got != c.want {
			t.Errorf("SplitShardFileName(%q, %d, %d) = %q, want %q", c.prefix, c.index, c.count, got, c.want)
		}
	}
}

func TestSplitShardNameRoundTrip(t *testing.T) {
	for _, name := range []string{
		"model-00001-of-00002.gguf",
		"model-00007-of-00042.gguf",
	} {
		prefix, index, count, ok := SplitShardName(name)
		if !ok {
			t.Fatalf("SplitShardName(%q) did not match", name)
		}
		if got := SplitShardFileName(prefix, index, count); got != name {
			t.Errorf("round trip = %q, want %q", got, name)
		}
	}
}
