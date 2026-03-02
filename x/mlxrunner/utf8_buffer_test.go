package mlxrunner

import (
	"bytes"
	"testing"
)

func TestFlushValidUTF8Prefix_PreservesIncompleteRune(t *testing.T) {
	var b bytes.Buffer

	b.Write([]byte{0xE3, 0x81})
	if got := flushValidUTF8Prefix(&b); got != "" {
		t.Fatalf("first flush = %q, want empty", got)
	}

	b.Write([]byte{0x93, 0xE3})
	if got := flushValidUTF8Prefix(&b); got != "こ" {
		t.Fatalf("second flush = %q, want %q", got, "こ")
	}

	if got := b.Bytes(); !bytes.Equal(got, []byte{0xE3}) {
		t.Fatalf("buffer after second flush = %v, want %v", got, []byte{0xE3})
	}

	b.Write([]byte{0x82, 0x93})
	if got := flushValidUTF8Prefix(&b); got != "ん" {
		t.Fatalf("third flush = %q, want %q", got, "ん")
	}

	if b.Len() != 0 {
		t.Fatalf("buffer not empty after third flush: %d", b.Len())
	}
}

func TestFlushValidUTF8Prefix_ValidText(t *testing.T) {
	var b bytes.Buffer
	b.WriteString("hello 世界")

	if got := flushValidUTF8Prefix(&b); got != "hello 世界" {
		t.Fatalf("flush = %q, want %q", got, "hello 世界")
	}

	if b.Len() != 0 {
		t.Fatalf("buffer not empty after flush: %d", b.Len())
	}
}
