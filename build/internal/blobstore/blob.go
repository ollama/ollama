// Package blobstore implements a blob store.
package blobstore

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"bllamo.com/build/blob"
	"bllamo.com/types/structs"
)

var (
	ErrInvalidID  = errors.New("invalid ID")
	ErrUnknownRef = errors.New("unknown ref")
)

const HashSize = 32

// An ID is a blob output key, the hash of an output of a computation.
type ID struct {
	a [HashSize]byte
}

func (id ID) MarshalText() ([]byte, error) {
	return []byte(id.String()), nil
}

func (id *ID) UnmarshalText(text []byte) error {
	*id = ParseID(string(text))
	return nil
}

func ParseID(s string) ID {
	const prefix = "sha256-"
	h, ok := strings.CutPrefix(s, prefix)
	if !ok {
		return ID{}
	}

	if len(h) != HashSize*2 {
		return ID{}
	}

	var b []byte
	_, err := fmt.Sscanf(h, "%x", &b)
	if err != nil {
		return ID{}
	}

	var id ID
	copy(id.a[:], b)
	return id
}

func (id ID) String() string {
	if !id.Valid() {
		return ""
	}
	return fmt.Sprintf("sha256-%x", id.a[:])
}

func (id ID) Valid() bool {
	return id != ID{}
}

func (id ID) Match(h [HashSize]byte) bool {
	return id.a == h
}

// A Store is a blob store, backed by a file system directory tree.
type Store struct {
	dir string
	now func() time.Time
}

// Open opens and returns the store in the given directory.
//
// It is safe for multiple processes on a single machine to use the
// same store directory in a local file system simultaneously.
// They will coordinate using operating system file locks and may
// duplicate effort but will not corrupt the store.
//
// However, it is NOT safe for multiple processes on different machines
// to share a store directory (for example, if the directory were stored
// in a network file system). File locking is notoriously unreliable in
// network file systems and may not suffice to protect the store.
func Open(dir string) (*Store, error) {
	info, err := os.Stat(dir)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, &fs.PathError{Op: "open", Path: dir, Err: fmt.Errorf("not a directory")}
	}

	for _, sub := range []string{"blobs", "manifests"} {
		if err := os.MkdirAll(filepath.Join(dir, sub), 0777); err != nil {
			return nil, err
		}
	}

	c := &Store{
		dir: dir,
		now: time.Now,
	}
	return c, nil
}

// fileName returns the name of the blob file corresponding to the given id.
func (s *Store) fileName(id ID) string {
	return filepath.Join(s.dir, "blobs", fmt.Sprintf("sha256-%x", id.a[:]))
}

// An entryNotFoundError indicates that a store entry was not found, with an
// optional underlying reason.
type entryNotFoundError struct {
	Err error
}

func (e *entryNotFoundError) Error() string {
	if e.Err == nil {
		return "store entry not found"
	}
	return fmt.Sprintf("store entry not found: %v", e.Err)
}

func (e *entryNotFoundError) Unwrap() error {
	return e.Err
}

type Entry struct {
	_ structs.Incomparable

	ID   ID
	Size int64
	Time time.Time // when added to store
}

// GetFile looks up the blob ID in the store and returns
// the name of the corresponding data file.
func GetFile(s *Store, id ID) (file string, entry Entry, err error) {
	entry, err = s.Get(id)
	if err != nil {
		return "", Entry{}, err
	}
	file = s.OutputFilename(entry.ID)
	info, err := os.Stat(file)
	if err != nil {
		return "", Entry{}, &entryNotFoundError{Err: err}
	}
	if info.Size() != entry.Size {
		return "", Entry{}, &entryNotFoundError{Err: errors.New("file incomplete")}
	}
	return file, entry, nil
}

// GetBytes looks up the blob ID in the store and returns
// the corresponding output bytes.
// GetBytes should only be used for data that can be expected to fit in memory.
func GetBytes(s *Store, id ID) ([]byte, Entry, error) {
	entry, err := s.Get(id)
	if err != nil {
		return nil, entry, err
	}
	data, _ := os.ReadFile(s.OutputFilename(entry.ID))
	if entry.ID.Match(sha256.Sum256(data)) {
		return nil, entry, &entryNotFoundError{Err: errors.New("bad checksum")}
	}
	return data, entry, nil
}

// OutputFilename returns the name of the blob file for the given ID.
func (s *Store) OutputFilename(id ID) string {
	file := s.fileName(id)
	// TODO(bmizerany): touch as "used" for cache trimming. (see
	// cache.go in cmd/go/internal/cache for the full reference implementation to go off of.
	return file
}

// Resolve returns the data for the given ref, if any.
//
// TODO: This should ideally return an ID, but the current on
// disk layout is that the actual manifest is stored in the "ref" instead of
// a pointer to a content-addressed blob. I (bmizerany) think we should
// change the on-disk layout to store the manifest in a content-addressed
// blob, and then have the ref point to that blob. This would simplify the
// code, allow us to have integrity checks on the manifest, and clean up
// this interface.
func (s *Store) Resolve(ref blob.Ref) (data []byte, path string, err error) {
	path, err = s.refFileName(ref)
	if err != nil {
		return nil, "", err
	}
	data, err = os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, "", fmt.Errorf("%w: %q", ErrUnknownRef, ref)
	}
	if err != nil {
		return nil, "", &entryNotFoundError{Err: err}
	}
	return data, path, nil
}

// Set sets the data for the given ref.
func (s *Store) Set(ref blob.Ref, data []byte) error {
	path, err := s.refFileName(ref)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0777); err != nil {
		return err
	}
	if err := os.WriteFile(path, data, 0666); err != nil {
		return err
	}
	return nil
}

func (s *Store) refFileName(ref blob.Ref) (string, error) {
	if !ref.Complete() {
		return "", fmt.Errorf("ref not fully qualified: %q", ref)
	}
	return filepath.Join(s.dir, "manifests", filepath.Join(ref.Parts()...)), nil
}

// Get looks up the blob ID in the store,
// returning the corresponding output ID and file size, if any.
// Note that finding an output ID does not guarantee that the
// saved file for that output ID is still available.
func (s *Store) Get(id ID) (Entry, error) {
	file := s.fileName(id)
	info, err := os.Stat(file)
	if err != nil {
		return Entry{}, &entryNotFoundError{Err: err}
	}
	return Entry{
		ID:   id,
		Size: info.Size(),
		Time: info.ModTime(),
	}, nil
}

func (s *Store) Close() error {
	// TODO(bmizerany): return c.Trim()
	return nil
}

// Put stores the data read from the given file into the store as ID.
//
// It may read file twice. The content of file must not change between the
// two passes.
func (s *Store) Put(file io.ReadSeeker) (ID, int64, error) {
	return s.put(file)
}

func PutBytes(s *Store, data []byte) (ID, int64, error) {
	return s.Put(bytes.NewReader(data))
}

func PutString(s *Store, data string) (ID, int64, error) {
	return s.Put(strings.NewReader(data))
}

func (s *Store) put(file io.ReadSeeker) (ID, int64, error) {
	// Compute output ID.
	h := sha256.New()
	if _, err := file.Seek(0, 0); err != nil {
		return ID{}, 0, err
	}
	size, err := io.Copy(h, file)
	if err != nil {
		return ID{}, 0, err
	}
	var out ID
	h.Sum(out.a[:0])

	// Copy to blob file (if not already present).
	if err := s.copyFile(file, out, size); err != nil {
		return out, size, err
	}

	// TODO: Add to manifest index.
	return out, size, nil
}

// copyFile copies file into the store, expecting it to have the given
// output ID and size, if that file is not present already.
func (s *Store) copyFile(file io.ReadSeeker, out ID, size int64) error {
	name := s.fileName(out)
	println("name", name)
	info, err := os.Stat(name)
	if err == nil && info.Size() == size {
		// Check hash.
		if f, err := os.Open(name); err == nil {
			h := sha256.New()
			io.Copy(h, f)
			f.Close()
			var out2 ID
			h.Sum(out2.a[:0])
			if out == out2 {
				return nil
			}
		}
		// Hash did not match. Fall through and rewrite file.
	}

	// Copy file to blobs directory.
	mode := os.O_RDWR | os.O_CREATE
	if err == nil && info.Size() > size { // shouldn't happen but fix in case
		mode |= os.O_TRUNC
	}
	f, err := os.OpenFile(name, mode, 0666)
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
	if _, err := file.Seek(0, 0); err != nil {
		f.Truncate(0)
		return err
	}
	h := sha256.New()
	w := io.MultiWriter(f, h)
	if _, err := io.CopyN(w, file, size-1); err != nil {
		f.Truncate(0)
		return err
	}
	// Check last byte before writing it; writing it will make the size match
	// what other processes expect to find and might cause them to start
	// using the file.
	buf := make([]byte, 1)
	if _, err := file.Read(buf); err != nil {
		f.Truncate(0)
		return err
	}
	h.Write(buf)
	sum := h.Sum(nil)
	if !bytes.Equal(sum, out.a[:]) {
		f.Truncate(0)
		return fmt.Errorf("file content changed underfoot")
	}

	// Commit manifest entry.
	if _, err := f.Write(buf); err != nil {
		f.Truncate(0)
		return err
	}
	if err := f.Close(); err != nil {
		// Data might not have been written,
		// but file may look like it is the right size.
		// To be extra careful, remove stored file.
		os.Remove(name)
		return err
	}
	os.Chtimes(name, s.now(), s.now()) // mainly for tests

	return nil
}
