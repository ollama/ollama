package blob

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"slices"
	"strings"
)

var ErrInvalidDigest = errors.New("invalid digest")

// RollingDigest is a rolling hash of a sequence of blobs. It tracks the last
// digest that was appended to it for comparisons. See
// [RollingDigest.CompareAndAppend].
//
// Its zero value is a valid RollingDigest that has not had any blobs appended
// to it.
type RollingDigest struct {
	h hash.Hash
}

// Current reports if the given digest matches the current rolling sum.
func (h *RollingDigest) Current() Digest {
	if h.h == nil {
		return Digest{}
	}
	var current [32]byte
	h.h.Sum(current[:0])
	return Digest{current}
}

// Append appends the given digest to the rolling sum in [Digest.String] form.
func (h *RollingDigest) Append(o Digest) {
	if h.h == nil {
		h.h = sha256.New()
	}
	fmt.Fprintf(h.h, "sha256:%x", o.sum)
}

// Digest is a blob identifier that is the SHA-256 hash of a blob's content.
//
// It is comparable and can be used as a map key.
type Digest struct {
	sum [32]byte
}

func (d *Digest) Sum() [32]byte {
	return d.sum
}

// ParseDigest parses a digest from a string. If the string is not a valid
// digest, a call to the returned digest's IsValid method will return false.
//
// The input string may be in one of two forms:
//
//   - ("sha256-<hex>"), where <hex> is a 64-character hexadecimal string.
//   - ("sha256:<hex>"), where <hex> is a 64-character hexadecimal string.
//
// The [Digest.String] method will return the canonical form of the
// digest, "sha256:<hex>".
func ParseDigest[S ~[]byte | ~string](v S) (Digest, error) {
	s := string(v)
	i := strings.IndexAny(s, ":-")
	var zero Digest
	if i < 0 {
		return zero, ErrInvalidDigest
	}

	prefix, sum := s[:i], s[i+1:]
	if prefix != "sha256" || len(sum) != 64 {
		return zero, ErrInvalidDigest
	}

	var d Digest
	_, err := hex.Decode(d.sum[:], []byte(sum))
	if err != nil {
		return zero, ErrInvalidDigest
	}
	return d, nil
}

func DigestFromBytes[S ~[]byte | ~string](v S) Digest {
	return Digest{sha256.Sum256([]byte(v))}
}

// String returns the string representation of the digest in the conventional
// form "sha256:<hex>".
func (d Digest) String() string {
	return fmt.Sprintf("sha256:%x", d.sum[:])
}

func (d Digest) Short() string {
	return fmt.Sprintf("%x", d.sum[:4])
}

func (d Digest) Compare(other Digest) int {
	return slices.Compare(d.sum[:], other.sum[:])
}

// IsValid returns true if the digest is valid, i.e. if it is the SHA-256 hash
// of some content.
func (d Digest) IsValid() bool {
	return d != (Digest{})
}

// MarshalText implements the encoding.TextMarshaler interface. It returns an
// error if [Digest.IsValid] returns false.
func (d Digest) MarshalText() ([]byte, error) {
	return []byte(d.String()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface, and only
// works for a zero digest. If [Digest.IsValid] returns true, it returns an
// error.
func (d *Digest) UnmarshalText(text []byte) error {
	if *d != (Digest{}) {
		return errors.New("digest: illegal UnmarshalText on valid digest")
	}
	v, err := ParseDigest(string(text))
	if err != nil {
		return err
	}
	*d = v
	return nil
}
