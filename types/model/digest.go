package model

import (
	"fmt"
	"log/slog"
	"strings"
	"unicode"
)

// Digest represents a digest of a model Manifest. It is a comparable value
// type and is immutable.
//
// The zero Digest is not a valid digest.
type Digest struct {
	s string
}

// Split returns the digest type and the digest value.
func (d Digest) Split() (typ, digest string) {
	typ, digest, _ = strings.Cut(d.s, "-")
	return
}

// String returns the digest in the form of "<digest-type>-<digest>", or the
// empty string if the digest is invalid.
func (d Digest) String() string { return d.s }

// IsValid returns true if the digest is valid (not zero).
//
// A valid digest may be created only by ParseDigest, or
// ParseName(name).Digest().
func (d Digest) IsValid() bool { return d.s != "" }

// LogValue implements slog.Value.
func (d Digest) LogValue() slog.Value {
	return slog.StringValue(d.String())
}

var (
	_ slog.LogValuer = Digest{}
)

// ParseDigest parses a string in the form of "<digest-type>-<digest>" into a
// Digest.
func ParseDigest(s string) Digest {
	typ, digest, ok := strings.Cut(s, "-")
	if !ok {
		typ, digest, ok = strings.Cut(s, ":")
	}
	if ok && isValidDigestType(typ) && isValidHex(digest) && len(digest) >= 2 {
		return Digest{s: fmt.Sprintf("%s-%s", typ, digest)}
	}
	return Digest{}
}

func MustParseDigest(s string) Digest {
	d := ParseDigest(s)
	if !d.IsValid() {
		panic(fmt.Sprintf("invalid digest: %q", s))
	}
	return d
}

func isValidDigestType(s string) bool {
	if len(s) == 0 {
		return false
	}
	for _, r := range s {
		if !unicode.IsLower(r) && !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

func isValidHex(s string) bool {
	if len(s) == 0 {
		return false
	}
	for i := range s {
		c := s[i]
		if c < '0' || c > '9' && c < 'a' || c > 'f' {
			return false
		}
	}
	return true
}
