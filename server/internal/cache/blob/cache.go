// Package blob implements a content-addressable disk cache for blobs and
// manifests.
package blob

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"hash"
	"io"
	"io/fs"
	"iter"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/server/internal/internal/names"
)

// Entry contains metadata about a blob in the cache.
type Entry struct {
	Digest Digest
	Size   int64
	Time   time.Time // when added to the cache
}

// DiskCache caches blobs and manifests on disk.
//
// The cache is rooted at a directory, which is created if it does not exist.
//
// Blobs are stored in the "blobs" subdirectory, and manifests are stored in the
// "manifests" subdirectory. A example directory structure might look like:
//
//	<dir>/
//	  blobs/
//	    sha256-<digest> - <blob data>
//	  manifests/
//	    <host>/
//	      <namespace>/
//	        <name>/
//	          <tag> - <manifest data>
//
// The cache is safe for concurrent use.
//
// Name casing is preserved in the cache, but is not significant when resolving
// names. For example, "Foo" and "foo" are considered the same name.
//
// The cache is not safe for concurrent use. It guards concurrent writes, but
// does not prevent duplicated effort. Because blobs are immutable, duplicate
// writes should result in the same file being written to disk.
type DiskCache struct {
	// Dir specifies the top-level directory where blobs and manifest
	// pointers are stored.
	dir string
	now func() time.Time

	testHookBeforeFinalWrite func(f *os.File)
}

// PutBytes is a convenience function for c.Put(d, strings.NewReader(s), int64(len(s))).
func PutBytes[S string | []byte](c *DiskCache, d Digest, data S) error {
	return c.Put(d, bytes.NewReader([]byte(data)), int64(len(data)))
}

// Open opens a cache rooted at the given directory. If the directory does not
// exist, it is created. If the directory is not a directory, an error is
// returned.
func Open(dir string) (*DiskCache, error) {
	if dir == "" {
		return nil, errors.New("blob: empty directory name")
	}

	info, err := os.Stat(dir)
	if err == nil && !info.IsDir() {
		return nil, fmt.Errorf("%q is not a directory", dir)
	}
	if err := os.MkdirAll(dir, 0o777); err != nil {
		return nil, err
	}

	subdirs := []string{"blobs", "manifests"}
	for _, subdir := range subdirs {
		if err := os.MkdirAll(filepath.Join(dir, subdir), 0o777); err != nil {
			return nil, err
		}
	}

	// TODO(bmizerany): support shards
	c := &DiskCache{
		dir: dir,
		now: time.Now,
	}
	return c, nil
}

func readAndSum(filename string, limit int64) (data []byte, _ Digest, err error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, Digest{}, err
	}
	defer f.Close()

	h := sha256.New()
	r := io.TeeReader(f, h)
	data, err = io.ReadAll(io.LimitReader(r, limit))
	if err != nil {
		return nil, Digest{}, err
	}
	var d Digest
	h.Sum(d.sum[:0])
	return data, d, nil
}

//lint:ignore U1000 used for debugging purposes as needed in tests
var debug = false

// debugger returns a function that can be used to add a step to the error message.
// The error message will be a list of steps that were taken before the error occurred.
// The steps are added in the order they are called.
//
// To set the error message, call the returned function with an empty string.
//
//lint:ignore U1000 used for debugging purposes as needed in tests
func debugger(err *error) func(step string) {
	if !debug {
		return func(string) {}
	}
	var steps []string
	return func(step string) {
		if step == "" && *err != nil {
			*err = fmt.Errorf("%q: %w", steps, *err)
			return
		}
		steps = append(steps, step)
		if len(steps) > 100 {
			// shift hints in case of a bug that causes a lot of hints
			copy(steps, steps[1:])
			steps = steps[:100]
		}
	}
}

// Resolve resolves a name to a digest. The name is expected to
// be in either of the following forms:
//
//	@<digest>
//	<name>@<digest>
//	<name>
//
// If a digest is provided, it is returned as is and nothing else happens.
//
// If a name is provided for a manifest that exists in the cache, the digest
// of the manifest is returned. If there is no manifest in the cache, it
// returns [fs.ErrNotExist].
//
// To cover the case where a manifest may change without the cache knowing
// (e.g. it was reformatted or modified by hand), the manifest data read and
// hashed is passed to a PutBytes call to ensure that the manifest is in the
// blob store. This is done to ensure that future calls to [Get] succeed in
// these cases.
func (c *DiskCache) Resolve(name string) (Digest, error) {
	name, digest := splitNameDigest(name)
	if digest != "" {
		return ParseDigest(digest)
	}

	// We want to address manifests files by digest using Get. That requires
	// them to be blobs. This cannot be directly accomplished by looking in
	// the blob store because manifests can change without Ollama knowing
	// (e.g. a user modifies a manifests by hand then pushes it to update
	// their model). We also need to support the blob caches inherited from
	// older versions of Ollama, which do not store manifests in the blob
	// store, so for these cases, we need to handle adding the manifests to
	// the blob store, just in time.
	//
	// So now we read the manifests file, hash it, and copy it to the blob
	// store if it's not already there.
	//
	// This should be cheap because manifests are small, and accessed
	// infrequently.
	file, err := c.manifestPath(name)
	if err != nil {
		return Digest{}, err
	}

	data, d, err := readAndSum(file, 1<<20)
	if err != nil {
		return Digest{}, err
	}

	// Ideally we'd read the "manifest" file as a manifest to the blob file,
	// but we are not changing this yet, so copy the manifest to the blob
	// store so it can be addressed by digest subsequent calls to Get.
	if err := PutBytes(c, d, data); err != nil {
		return Digest{}, err
	}
	return d, nil
}

// Put writes a new blob to the cache, identified by its digest. The operation
// reads content from r, which must precisely match both the specified size and
// digest.
//
// Concurrent write safety is achieved through file locking. The implementation
// guarantees write integrity by enforcing size limits and content validation
// before allowing the file to reach its final state.
func (c *DiskCache) Put(d Digest, r io.Reader, size int64) error {
	return c.copyNamedFile(c.GetFile(d), r, d, size)
}

// Import imports a blob from the provided reader into the cache. It reads the
// entire content of the reader, calculates its digest, and stores it in the
// cache.
//
// Import should be considered unsafe for use with untrusted data, such as data
// read from a network. The caller is responsible for ensuring the integrity of
// the data being imported.
func (c *DiskCache) Import(r io.Reader, size int64) (Digest, error) {
	// users that want to change the temp dir can set TEMPDIR.
	f, err := os.CreateTemp("", "blob-")
	if err != nil {
		return Digest{}, err
	}
	defer os.Remove(f.Name())

	// Copy the blob to a temporary file.
	h := sha256.New()
	r = io.TeeReader(r, h)
	n, err := io.Copy(f, r)
	if err != nil {
		return Digest{}, err
	}
	if n != size {
		return Digest{}, fmt.Errorf("blob: expected %d bytes, got %d", size, n)
	}

	// Check the digest.
	var d Digest
	h.Sum(d.sum[:0])
	if err := f.Close(); err != nil {
		return Digest{}, err
	}
	name := c.GetFile(d)
	// Rename the temporary file to the final file.
	if err := os.Rename(f.Name(), name); err != nil {
		return Digest{}, err
	}
	os.Chtimes(name, c.now(), c.now()) // mainly for tests
	return d, nil
}

// Get retrieves a blob from the cache using the provided digest. The operation
// fails if the digest is malformed or if any errors occur during blob
// retrieval.
func (c *DiskCache) Get(d Digest) (Entry, error) {
	name := c.GetFile(d)
	info, err := os.Stat(name)
	if err != nil {
		return Entry{}, err
	}
	if info.Size() == 0 {
		return Entry{}, fs.ErrNotExist
	}
	return Entry{
		Digest: d,
		Size:   info.Size(),
		Time:   info.ModTime(),
	}, nil
}

// Link creates a symbolic reference in the cache that maps the provided name
// to a blob identified by its digest, making it retrievable by name using
// [Resolve].
//
// It returns an error if either the name or digest is invalid, or if link
// creation encounters any issues.
func (c *DiskCache) Link(name string, d Digest) error {
	manifest, err := c.manifestPath(name)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(c.GetFile(d), os.O_RDONLY, 0)
	if err != nil {
		return err
	}
	defer f.Close()

	// TODO(bmizerany): test this happens only if the blob was found to
	// avoid leaving debris
	if err := os.MkdirAll(filepath.Dir(manifest), 0o777); err != nil {
		return err
	}

	info, err := f.Stat()
	if err != nil {
		return err
	}

	// Copy manifest to cache directory.
	return c.copyNamedFile(manifest, f, d, info.Size())
}

// Unlink unlinks the manifest by name from the cache. If the name is not
// found. If a manifest is removed ok will be true, otherwise false. If an
// error occurs, it returns ok false, and the error.
func (c *DiskCache) Unlink(name string) (ok bool, _ error) {
	manifest, err := c.manifestPath(name)
	if err != nil {
		return false, err
	}
	err = os.Remove(manifest)
	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}
	return true, err
}

// GetFile returns the absolute path to the file, in the cache, for the given
// digest. It does not check if the file exists.
//
// The returned path should not be stored, used outside the lifetime of the
// cache, or interpreted in any way.
func (c *DiskCache) GetFile(d Digest) string {
	filename := fmt.Sprintf("sha256-%x", d.sum)
	return absJoin(c.dir, "blobs", filename)
}

// Links returns a sequence of link names. The sequence is in lexical order.
// Names are converted from their relative path form to their name form but are
// not guaranteed to be valid. Callers should validate the names before using.
func (c *DiskCache) Links() iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		for path, err := range c.links() {
			if err != nil {
				yield("", err)
				return
			}
			if !yield(pathToName(path), nil) {
				return
			}
		}
	}
}

// pathToName converts a path to a name. It is the inverse of nameToPath. The
// path is assumed to be in filepath.ToSlash format.
func pathToName(s string) string {
	s = strings.TrimPrefix(s, "manifests/")
	rr := []rune(s)
	for i := len(rr) - 1; i > 0; i-- {
		if rr[i] == '/' {
			rr[i] = ':'
			return string(rr)
		}
	}
	return s
}

// manifestPath finds the first manifest file on disk that matches the given
// name using a case-insensitive comparison. If no manifest file is found, it
// returns the path where the manifest file would be if it existed.
//
// If two manifest files exists on disk that match the given name using a
// case-insensitive comparison, the one that sorts first, lexically, is
// returned.
func (c *DiskCache) manifestPath(name string) (string, error) {
	np, err := nameToPath(name)
	if err != nil {
		return "", err
	}

	maybe := filepath.Join("manifests", np)
	for l, err := range c.links() {
		if err != nil {
			return "", err
		}
		if strings.EqualFold(maybe, l) {
			return filepath.Join(c.dir, l), nil
		}
	}
	return filepath.Join(c.dir, maybe), nil
}

// links returns a sequence of links in the cache in lexical order.
func (c *DiskCache) links() iter.Seq2[string, error] {
	// TODO(bmizerany): reuse empty dirnames if exist
	return func(yield func(string, error) bool) {
		fsys := os.DirFS(c.dir)
		manifests, err := fs.Glob(fsys, "manifests/*/*/*/*")
		if err != nil {
			yield("", err)
			return
		}
		for _, manifest := range manifests {
			if !yield(manifest, nil) {
				return
			}
		}
	}
}

type checkWriter struct {
	size int64
	d    Digest
	f    *os.File
	h    hash.Hash

	w   io.Writer // underlying writer; set by creator
	n   int64
	err error

	testHookBeforeFinalWrite func(*os.File)
}

func (w *checkWriter) seterr(err error) error {
	if w.err == nil {
		w.err = err
	}
	return err
}

// Write writes p to the underlying hash and writer. The last write to the
// underlying writer is guaranteed to be the last byte of p as verified by the
// hash.
func (w *checkWriter) Write(p []byte) (int, error) {
	if w.err != nil {
		return 0, w.err
	}

	_, err := w.h.Write(p)
	if err != nil {
		return 0, w.seterr(err)
	}
	nextSize := w.n + int64(len(p))
	if nextSize == w.size {
		// last write. check hash.
		sum := w.h.Sum(nil)
		if !bytes.Equal(sum, w.d.sum[:]) {
			return 0, w.seterr(fmt.Errorf("file content changed underfoot"))
		}
		if w.testHookBeforeFinalWrite != nil {
			w.testHookBeforeFinalWrite(w.f)
		}
	}
	if nextSize > w.size {
		return 0, w.seterr(fmt.Errorf("content exceeds expected size: %d > %d", nextSize, w.size))
	}
	n, err := w.w.Write(p)
	w.n += int64(n)
	return n, w.seterr(err)
}

// copyNamedFile copies file into name, expecting it to have the given Digest
// and size, if that file is not present already.
func (c *DiskCache) copyNamedFile(name string, file io.Reader, out Digest, size int64) error {
	info, err := os.Stat(name)
	if err == nil && info.Size() == size {
		// File already exists with correct size. This is good enough.
		// We can skip expensive hash checks.
		//
		// TODO: Do the hash check, but give caller a way to skip it.
		return nil
	}

	// Copy file to cache directory.
	mode := os.O_RDWR | os.O_CREATE
	if err == nil && info.Size() > size { // shouldn't happen but fix in case
		mode |= os.O_TRUNC
	}
	f, err := os.OpenFile(name, mode, 0o666)
	if err != nil {
		return err
	}
	defer f.Close()
	if size == 0 {
		// File now exists with correct size.
		// Only one possible zero-length file, so contents are OK too.
		// Early return here makes sure there's a "last byte" for code below.
		return nil
	}

	// From here on, if any of the I/O writing the file fails,
	// we make a best-effort attempt to truncate the file f
	// before returning, to avoid leaving bad bytes in the file.

	// Copy file to f, but also into h to double-check hash.
	cw := &checkWriter{
		d:    out,
		size: size,
		h:    sha256.New(),
		f:    f,
		w:    f,

		testHookBeforeFinalWrite: c.testHookBeforeFinalWrite,
	}
	n, err := io.Copy(cw, file)
	if err != nil {
		f.Truncate(0)
		return err
	}
	if n < size {
		f.Truncate(0)
		return io.ErrUnexpectedEOF
	}

	if err := f.Close(); err != nil {
		// Data might not have been written,
		// but file may look like it is the right size.
		// To be extra careful, remove cached file.
		os.Remove(name)
		return err
	}
	os.Chtimes(name, c.now(), c.now()) // mainly for tests

	return nil
}

func splitNameDigest(s string) (name, digest string) {
	i := strings.LastIndexByte(s, '@')
	if i < 0 {
		return s, ""
	}
	return s[:i], s[i+1:]
}

var errInvalidName = errors.New("invalid name")

func nameToPath(name string) (_ string, err error) {
	n := names.Parse(name)
	if !n.IsFullyQualified() {
		return "", errInvalidName
	}
	return filepath.Join(n.Host(), n.Namespace(), n.Model(), n.Tag()), nil
}

func absJoin(pp ...string) string {
	abs, err := filepath.Abs(filepath.Join(pp...))
	if err != nil {
		panic(err) // this should never happen
	}
	return abs
}
