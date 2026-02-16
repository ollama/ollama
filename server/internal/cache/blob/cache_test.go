package blob

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/server/internal/testutil"
)

func init() {
	debug = true
}

var epoch = func() time.Time {
	d := time.Date(2021, 1, 1, 0, 0, 0, 0, time.UTC)
	if d.IsZero() {
		panic("time zero")
	}
	return d
}()

func TestOpenErrors(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		panic(err)
	}

	cases := []struct {
		dir string
		err string
	}{
		{t.TempDir(), ""},
		{"", "empty directory name"},
		{exe, "not a directory"},
	}

	for _, tt := range cases {
		t.Run(tt.dir, func(t *testing.T) {
			_, err := Open(tt.dir)
			if tt.err == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.err) {
				t.Fatalf("err = %v, want %q", err, tt.err)
			}
		})
	}
}

func TestGetFile(t *testing.T) {
	t.Chdir(t.TempDir())

	c, err := Open(".")
	if err != nil {
		t.Fatal(err)
	}

	d := mkdigest("1")
	got := c.GetFile(d)
	cleaned := filepath.Clean(got)
	if cleaned != got {
		t.Fatalf("got is unclean: %q", got)
	}
	if !filepath.IsAbs(got) {
		t.Fatal("got is not absolute")
	}
	abs, _ := filepath.Abs(c.dir)
	if !strings.HasPrefix(got, abs) {
		t.Fatalf("got is not local to %q", c.dir)
	}
}

func TestBasic(t *testing.T) {
	c, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	now := epoch
	c.now = func() time.Time { return now }

	checkEntry := entryChecker(t, c)
	checkFailed := func(err error) {
		if err == nil {
			t.Helper()
			t.Fatal("expected error")
		}
	}

	_, err = c.Resolve("invalid")
	checkFailed(err)

	_, err = c.Resolve("h/n/m:t")
	checkFailed(err)

	dx := mkdigest("x")

	d, err := c.Resolve(fmt.Sprintf("h/n/m:t@%s", dx))
	if err != nil {
		t.Fatal(err)
	}
	if d != dx {
		t.Fatalf("d = %v, want %v", d, dx)
	}

	_, err = c.Get(Digest{})
	checkFailed(err)

	// not committed yet
	_, err = c.Get(dx)
	checkFailed(err)

	err = PutBytes(c, dx, "!")
	checkFailed(err)

	err = PutBytes(c, dx, "x")
	if err != nil {
		t.Fatal(err)
	}
	checkEntry(dx, 1, now)

	t0 := now
	now = now.Add(1*time.Hour + 1*time.Minute)
	err = PutBytes(c, dx, "x")
	if err != nil {
		t.Fatal(err)
	}

	// check not updated
	checkEntry(dx, 1, t0)
}

type sleepFunc func(d time.Duration) time.Time

func openTester(t *testing.T) (*DiskCache, sleepFunc) {
	t.Helper()
	c, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	now := epoch
	c.now = func() time.Time { return now }
	return c, func(d time.Duration) time.Time {
		now = now.Add(d)
		return now
	}
}

func TestManifestPath(t *testing.T) {
	check := testutil.Checker(t)

	c, sleep := openTester(t)

	d1 := mkdigest("1")
	err := PutBytes(c, d1, "1")
	check(err)

	err = c.Link("h/n/m:t", d1)
	check(err)

	t0 := sleep(0)
	sleep(1 * time.Hour)
	err = c.Link("h/n/m:t", d1) // nop expected
	check(err)

	file := must(c.manifestPath("h/n/m:t"))
	info, err := os.Stat(file)
	check(err)
	testutil.CheckTime(t, info.ModTime(), t0)
}

func TestManifestExistsWithoutBlob(t *testing.T) {
	t.Chdir(t.TempDir())

	check := testutil.Checker(t)

	c, err := Open(".")
	check(err)

	checkEntry := entryChecker(t, c)

	man := must(c.manifestPath("h/n/m:t"))
	os.MkdirAll(filepath.Dir(man), 0o777)
	testutil.WriteFile(t, man, "1")

	got, err := c.Resolve("h/n/m:t")
	check(err)

	want := mkdigest("1")
	if got != want {
		t.Fatalf("got = %v, want %v", got, want)
	}

	e, err := c.Get(got)
	check(err)
	checkEntry(got, 1, e.Time)
}

func TestPut(t *testing.T) {
	c, sleep := openTester(t)

	check := testutil.Checker(t)
	checkEntry := entryChecker(t, c)

	d := mkdigest("hello, world")
	err := PutBytes(c, d, "hello")
	if err == nil {
		t.Fatal("expected error")
	}

	got, err := c.Get(d)
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("expected error, got %v", got)
	}

	// Put a valid blob
	err = PutBytes(c, d, "hello, world")
	check(err)
	checkEntry(d, 12, sleep(0))

	// Put a blob with content that does not hash to the digest
	err = PutBytes(c, d, "hello")
	if err == nil {
		t.Fatal("expected error")
	}
	checkNotExists(t, c, d)

	// Put the valid blob back and check it
	err = PutBytes(c, d, "hello, world")
	check(err)
	checkEntry(d, 12, sleep(0))

	// Put a blob that errors during Read
	err = c.Put(d, &errOnBangReader{s: "!"}, 1)
	if err == nil {
		t.Fatal("expected error")
	}
	checkNotExists(t, c, d)

	// Put valid blob back and check it
	err = PutBytes(c, d, "hello, world")
	check(err)
	checkEntry(d, 12, sleep(0))

	// Put a blob with mismatched size
	err = c.Put(d, strings.NewReader("hello, world"), 11)
	if err == nil {
		t.Fatal("expected error")
	}
	checkNotExists(t, c, d)

	// Final byte does not match the digest (testing commit phase)
	err = PutBytes(c, d, "hello, world$")
	if err == nil {
		t.Fatal("expected error")
	}
	checkNotExists(t, c, d)

	reset := c.setTestHookBeforeFinalWrite(func(f *os.File) {
		// change mode to read-only
		f.Truncate(0)
		f.Chmod(0o400)
		f.Close()
		f1, err := os.OpenFile(f.Name(), os.O_RDONLY, 0)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { f1.Close() })
		*f = *f1
	})
	defer reset()

	err = PutBytes(c, d, "hello, world")
	if err == nil {
		t.Fatal("expected error")
	}
	checkNotExists(t, c, d)
	reset()
}

func TestImport(t *testing.T) {
	c, _ := openTester(t)

	checkEntry := entryChecker(t, c)

	want := mkdigest("x")
	got, err := c.Import(strings.NewReader("x"), 1)
	if err != nil {
		t.Fatal(err)
	}
	if want != got {
		t.Fatalf("digest = %v, want %v", got, want)
	}
	checkEntry(want, 1, epoch)

	got, err = c.Import(strings.NewReader("x"), 1)
	if err != nil {
		t.Fatal(err)
	}
	if want != got {
		t.Fatalf("digest = %v, want %v", got, want)
	}
	checkEntry(want, 1, epoch)
}

func (c *DiskCache) setTestHookBeforeFinalWrite(h func(*os.File)) (reset func()) {
	old := c.testHookBeforeFinalWrite
	c.testHookBeforeFinalWrite = h
	return func() { c.testHookBeforeFinalWrite = old }
}

func TestPutGetZero(t *testing.T) {
	c, sleep := openTester(t)

	check := testutil.Checker(t)
	checkEntry := entryChecker(t, c)

	d := mkdigest("x")
	err := PutBytes(c, d, "x")
	check(err)
	checkEntry(d, 1, sleep(0))

	err = os.Truncate(c.GetFile(d), 0)
	check(err)

	_, err = c.Get(d)
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("err = %v, want fs.ErrNotExist", err)
	}
}

func TestPutZero(t *testing.T) {
	c, _ := openTester(t)
	d := mkdigest("x")
	err := c.Put(d, strings.NewReader("x"), 0) // size == 0 (not size of content)
	testutil.Check(t, err)
	checkNotExists(t, c, d)
}

func TestCommit(t *testing.T) {
	check := testutil.Checker(t)

	c, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	checkEntry := entryChecker(t, c)

	now := epoch
	c.now = func() time.Time { return now }

	d1 := mkdigest("1")
	err = c.Link("h/n/m:t", d1)
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("err = %v, want fs.ErrNotExist", err)
	}

	err = PutBytes(c, d1, "1")
	check(err)

	err = c.Link("h/n/m:t", d1)
	check(err)

	got, err := c.Resolve("h/n/m:t")
	check(err)
	if got != d1 {
		t.Fatalf("d = %v, want %v", got, d1)
	}

	// commit again, more than 1 byte
	d2 := mkdigest("22")
	err = PutBytes(c, d2, "22")
	check(err)
	err = c.Link("h/n/m:t", d2)
	check(err)
	checkEntry(d2, 2, now)

	filename := must(c.manifestPath("h/n/m:t"))
	data, err := os.ReadFile(filename)
	check(err)
	if string(data) != "22" {
		t.Fatalf("data = %q, want %q", data, "22")
	}

	t0 := now
	now = now.Add(1 * time.Hour)
	err = c.Link("h/n/m:t", d2) // same contents; nop
	check(err)
	info, err := os.Stat(filename)
	check(err)
	testutil.CheckTime(t, info.ModTime(), t0)
}

func TestManifestInvalidBlob(t *testing.T) {
	c, _ := openTester(t)
	d := mkdigest("1")
	err := c.Link("h/n/m:t", d)
	if err == nil {
		t.Fatal("expected error")
	}
	checkNotExists(t, c, d)

	err = PutBytes(c, d, "1")
	testutil.Check(t, err)
	err = os.WriteFile(c.GetFile(d), []byte("invalid"), 0o666)
	if err != nil {
		t.Fatal(err)
	}

	err = c.Link("h/n/m:t", d)
	if !strings.Contains(err.Error(), "underfoot") {
		t.Fatalf("err = %v, want error to contain %q", err, "underfoot")
	}
}

func TestManifestNameReuse(t *testing.T) {
	t.Run("case-insensitive", func(t *testing.T) {
		// This should run on all file system types.
		testManifestNameReuse(t)
	})
	t.Run("case-sensitive", func(t *testing.T) {
		useCaseInsensitiveTempDir(t)
		testManifestNameReuse(t)
	})
}

func testManifestNameReuse(t *testing.T) {
	check := testutil.Checker(t)

	c, _ := openTester(t)

	d1 := mkdigest("1")
	err := PutBytes(c, d1, "1")
	check(err)
	err = c.Link("h/n/m:t", d1)
	check(err)

	d2 := mkdigest("22")
	err = PutBytes(c, d2, "22")
	check(err)
	err = c.Link("H/N/M:T", d2)
	check(err)

	var g [2]Digest
	g[0], err = c.Resolve("h/n/m:t")
	check(err)
	g[1], err = c.Resolve("H/N/M:T")
	check(err)

	w := [2]Digest{d2, d2}
	if g != w {
		t.Fatalf("g = %v, want %v", g, w)
	}

	var got []string
	for l, err := range c.links() {
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, l)
	}
	want := []string{"manifests/h/n/m/t"}
	if !slices.Equal(got, want) {
		t.Fatalf("got = %v, want %v", got, want)
	}

	// relink with different case
	unlinked, err := c.Unlink("h/n/m:t")
	check(err)
	if !unlinked {
		t.Fatal("expected unlinked")
	}
	err = c.Link("h/n/m:T", d1)
	check(err)

	got = got[:0]
	for l, err := range c.links() {
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, l)
	}

	// we should have only one link that is same case as the last link
	want = []string{"manifests/h/n/m/T"}
	if !slices.Equal(got, want) {
		t.Fatalf("got = %v, want %v", got, want)
	}
}

func TestManifestFile(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"", ""},

		// valid names
		{"h/n/m:t", "/manifests/h/n/m/t"},
		{"hh/nn/mm:tt", "/manifests/hh/nn/mm/tt"},

		{"%/%/%/%", ""},

		// already a path
		{"h/n/m/t", ""},

		// refs are not names
		{"h/n/m:t@sha256-1", ""},
		{"m@sha256-1", ""},
		{"n/m:t@sha256-1", ""},
	}

	c, _ := openTester(t)
	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			got, err := c.manifestPath(tt.in)
			if err != nil && tt.want != "" {
				t.Fatalf("unexpected error: %v", err)
			}
			if err == nil && tt.want == "" {
				t.Fatalf("expected error")
			}
			dir := filepath.ToSlash(c.dir)
			got = filepath.ToSlash(got)
			got = strings.TrimPrefix(got, dir)
			if got != tt.want {
				t.Fatalf("got = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestNames(t *testing.T) {
	c, _ := openTester(t)
	check := testutil.Checker(t)

	check(PutBytes(c, mkdigest("1"), "1"))
	check(PutBytes(c, mkdigest("2"), "2"))

	check(c.Link("h/n/m:t", mkdigest("1")))
	check(c.Link("h/n/m:u", mkdigest("2")))

	var got []string
	for l, err := range c.Links() {
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, l)
	}
	want := []string{"h/n/m:t", "h/n/m:u"}
	if !slices.Equal(got, want) {
		t.Fatalf("got = %v, want %v", got, want)
	}
}

func mkdigest(s string) Digest {
	return Digest{sha256.Sum256([]byte(s))}
}

func checkNotExists(t *testing.T, c *DiskCache, d Digest) {
	t.Helper()
	_, err := c.Get(d)
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("err = %v, want fs.ErrNotExist", err)
	}
}

func entryChecker(t *testing.T, c *DiskCache) func(Digest, int64, time.Time) {
	t.Helper()
	return func(d Digest, size int64, mod time.Time) {
		t.Helper()
		t.Run("checkEntry:"+d.String(), func(t *testing.T) {
			t.Helper()

			defer func() {
				if t.Failed() {
					dumpCacheContents(t, c)
				}
			}()

			e, err := c.Get(d)
			if size == 0 && errors.Is(err, fs.ErrNotExist) {
				err = nil
			}
			if err != nil {
				t.Fatal(err)
			}
			if e.Digest != d {
				t.Errorf("e.Digest = %v, want %v", e.Digest, d)
			}
			if e.Size != size {
				t.Fatalf("e.Size = %v, want %v", e.Size, size)
			}

			testutil.CheckTime(t, e.Time, mod)
			info, err := os.Stat(c.GetFile(d))
			if err != nil {
				t.Fatal(err)
			}
			if info.Size() != size {
				t.Fatalf("info.Size = %v, want %v", info.Size(), size)
			}
			testutil.CheckTime(t, info.ModTime(), mod)
		})
	}
}

func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func TestNameToPath(t *testing.T) {
	_, err := nameToPath("h/n/m:t")
	if err != nil {
		t.Fatal(err)
	}
}

type errOnBangReader struct {
	s string
	n int
}

func (e *errOnBangReader) Read(p []byte) (int, error) {
	if len(p) < 1 {
		return 0, io.ErrShortBuffer
	}
	if e.n >= len(p) {
		return 0, io.EOF
	}
	if e.s[e.n] == '!' {
		return 0, errors.New("bang")
	}
	p[0] = e.s[e.n]
	e.n++
	return 1, nil
}

func dumpCacheContents(t *testing.T, c *DiskCache) {
	t.Helper()

	var b strings.Builder
	fsys := os.DirFS(c.dir)
	fs.WalkDir(fsys, ".", func(path string, d fs.DirEntry, err error) error {
		t.Helper()

		if err != nil {
			return err
		}
		info, err := d.Info()
		if err != nil {
			return err
		}

		// Format like ls:
		//
		// ; ls -la
		// drwxr-xr-x  224 Jan 13 14:22 blob/sha256-123
		// drwxr-xr-x  224 Jan 13 14:22 manifest/h/n/m

		fmt.Fprintf(&b, "    %s % 4d %s %s\n",
			info.Mode(),
			info.Size(),
			info.ModTime().Format("Jan 2 15:04"),
			path,
		)
		return nil
	})
	t.Log()
	t.Logf("cache contents:\n%s", b.String())
}
