package blobstore

import (
	"errors"
	"iter"
	"os"
	"path/filepath"
	"testing"
	"time"

	"bllamo.com/build/blob"
	"kr.dev/diff"
)

const (
	blobNameHello = "sha256-2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
)

func TestStoreBasicBlob(t *testing.T) {
	dir := t.TempDir()

	checkDir(t, dir, nil)

	st, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}

	now := time.Now()
	st.now = func() time.Time { return now }

	checkDir(t, dir, []string{
		"blobs/",
		"manifests/",
	})

	id, size, err := PutBytes(st, []byte("hello"))
	if err != nil {
		t.Fatal(err)
	}

	if id != ParseID(blobNameHello) {
		t.Errorf("unexpected ID: %s", id)
	}
	if size != 5 {
		t.Errorf("unexpected size: %d", size)
	}

	checkDir(t, dir, []string{
		"blobs/",
		"blobs/" + blobNameHello,
		"manifests/",
	})

	got, err := st.Get(id)
	if err != nil {
		t.Fatal(err)
	}

	diff.Test(t, t.Errorf, got, Entry{
		ID:   id,
		Size: 5,
		Time: now,
	})

	file := st.OutputFilename(id)
	wantFile := filepath.Join(dir, "blobs", blobNameHello)
	if file != wantFile {
		t.Errorf("unexpected file: %s", file)
	}

	// Check tags
	ref := blob.ParseRef("test+KQED")

	t.Logf("resolving %s", ref)

	data, _, err := st.Resolve(ref)
	var e *entryNotFoundError
	if !errors.As(err, &e) {
		t.Fatal(err)
	}
	if data != nil {
		t.Errorf("unexpected data: %q", data)
	}

	if err := st.Set(ref, []byte("{}")); err != nil {
		t.Fatal(err)
	}

	data, _, err = st.Resolve(ref)
	if err != nil {
		t.Fatal(err)
	}

	if g := string(data); g != "{}" {
		t.Errorf("g = %q; want %q", g, "{}")
	}

	checkDir(t, dir, []string{
		"blobs/",
		"blobs/sha256-2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
		"manifests/",
		"manifests/registry.ollama.ai/",
		"manifests/registry.ollama.ai/library/",
		"manifests/registry.ollama.ai/library/test/",
		"manifests/registry.ollama.ai/library/test/latest/",
		"manifests/registry.ollama.ai/library/test/latest/KQED",
	})
}

// checkDir checks that the directory at dir contains the files in want. The
// files in want must be relative to dir.
//
// direcotories are suffixed with a slash (e.g. "foo/" instead of "foo").
//
// want must be in lexicographic order.
func checkDir(t testing.TB, dir string, want []string) {
	t.Helper()

	var matches []string
	for path, err := range walkDir(dir) {
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("found %s", path)
		if path == "./" {
			continue
		}
		path = filepath.ToSlash(path)
		matches = append(matches, path)
	}

	diff.Test(t, t.Errorf, matches, want)
}

var errStop = errors.New("stop")

func walkDir(dir string) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		err := filepath.WalkDir(dir, func(path string, info os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			path, err = filepath.Rel(dir, path)
			if err != nil {
				return err
			}
			path = filepath.ToSlash(path)
			if info.IsDir() {
				path += "/"
			}
			if !yield(path, nil) {
				return errStop
			}
			return nil
		})
		if !errors.Is(err, errStop) && err != nil {
			yield("", err)
		}
	}
}
