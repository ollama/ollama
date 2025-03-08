package bufioutil

import (
	"bytes"
	"io"
	"strings"
	"testing"
)

func TestBufferedSeeker(t *testing.T) {
	const alphabet = "abcdefghijklmnopqrstuvwxyz"

	bs := NewBufferedSeeker(strings.NewReader(alphabet), 0) // minReadBufferSize = 16

	checkRead := func(buf []byte, expected string) {
		t.Helper()
		_, err := bs.Read(buf)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(buf, []byte(expected)) {
			t.Fatalf("expected %s, got %s", expected, buf)
		}
	}

	// Read the first 5 bytes
	buf := make([]byte, 5)

	checkRead(buf, "abcde")

	// Seek back to the beginning
	_, err := bs.Seek(0, io.SeekStart)
	if err != nil {
		t.Fatal(err)
	}

	// read 'a'
	checkRead(buf[:1], "a")

	if bs.br.Buffered() == 0 {
		t.Fatalf("totally unexpected sanity check failed")
	}

	// Seek past 'b'
	_, err = bs.Seek(1, io.SeekCurrent)
	if err != nil {
		t.Fatal(err)
	}
	checkRead(buf, "cdefg")

	// Seek back to the beginning
	_, err = bs.Seek(0, io.SeekStart)
	if err != nil {
		t.Fatal(err)
	}
	checkRead(buf, "abcde")

	// Seek to the end
	_, err = bs.Seek(-5, io.SeekEnd)
	if err != nil {
		t.Fatal(err)
	}
	checkRead(buf, "vwxyz")
}
