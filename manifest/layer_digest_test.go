package manifest

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"testing"
)

func TestNewLayerWithDigest(t *testing.T) {
	body := bytes.Repeat([]byte("gguf"), 4096)
	sum := sha256.Sum256(body)
	digest := fmt.Sprintf("sha256:%x", sum)

	t.Run("stores matching content", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())

		layer, err := NewLayerWithDigest(bytes.NewReader(body), MediaTypeImageModel, digest)
		if err != nil {
			t.Fatal(err)
		}
		if layer.Digest != digest || layer.Size != int64(len(body)) {
			t.Fatalf("layer = %+v, want digest %s and size %d", layer, digest, len(body))
		}

		p, err := BlobsPath(digest)
		if err != nil {
			t.Fatal(err)
		}
		got, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, body) {
			t.Fatalf("blob content does not match the upload")
		}
	})

	t.Run("leaves nothing behind on a mismatch", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())

		_, err := NewLayerWithDigest(bytes.NewReader(body), MediaTypeImageModel, "sha256:"+fmt.Sprintf("%064d", 0))
		if !errors.Is(err, ErrDigestMismatch) {
			t.Fatalf("err = %v, want ErrDigestMismatch", err)
		}

		blobs, err := BlobsPath("")
		if err != nil {
			t.Fatal(err)
		}
		entries, err := os.ReadDir(blobs)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 0 {
			t.Fatalf("blobs dir has %d entries after a rejected upload, want 0", len(entries))
		}
	})

	t.Run("leaves nothing behind when the body fails", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())

		if _, err := NewLayerWithDigest(io.MultiReader(bytes.NewReader(body), failReader{}), MediaTypeImageModel, digest); err == nil {
			t.Fatal("want an error for a truncated body")
		}

		blobs, _ := BlobsPath("")
		entries, err := os.ReadDir(blobs)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 0 {
			t.Fatalf("blobs dir has %d entries after a failed upload, want 0", len(entries))
		}
	})

	t.Run("keeps an existing copy intact", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())

		if _, err := NewLayerWithDigest(bytes.NewReader(body), MediaTypeImageModel, digest); err != nil {
			t.Fatal(err)
		}

		other := []byte("first")
		otherSum := sha256.Sum256(other)
		otherDigest := fmt.Sprintf("sha256:%x", otherSum)

		p, err := BlobsPath(otherDigest)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, other, 0o644); err != nil {
			t.Fatal(err)
		}

		// the digest already matches the content, so the existing file must win
		if _, err := NewLayerWithDigest(bytes.NewReader(other), MediaTypeImageModel, otherDigest); err != nil {
			t.Fatal(err)
		}
		got, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, other) {
			t.Fatalf("existing blob was replaced")
		}
	})
}

type failReader struct{}

func (failReader) Read([]byte) (int, error) { return 0, errors.New("connection reset") }
