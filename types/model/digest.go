package model

import (
	"database/sql"
	"database/sql/driver"
	"errors"
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

// Type returns the digest type of the digest.
//
// Example:
//
//	ParseDigest("sha256-1234").Type() // returns "sha256"
func (d Digest) Type() string {
	typ, _, _ := strings.Cut(d.s, "-")
	return typ
}

// String returns the digest in the form of "<digest-type>-<digest>", or the
// empty string if the digest is invalid.
func (d Digest) String() string { return d.s }

// IsValid returns true if the digest is valid (not zero).
//
// A valid digest may be created only by ParseDigest, or
// ParseName(name).Digest().
func (d Digest) IsValid() bool { return d.s != "" }

// MarshalText implements encoding.TextMarshaler.
func (d Digest) MarshalText() ([]byte, error) {
	return []byte(d.String()), nil
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (d *Digest) UnmarshalText(text []byte) error {
	if d.IsValid() {
		return errors.New("model.Digest: illegal UnmarshalText on valid Digest")
	}
	*d = ParseDigest(string(text))
	return nil
}

// LogValue implements slog.Value.
func (d Digest) LogValue() slog.Value {
	return slog.StringValue(d.String())
}

var (
	_ driver.Valuer  = Digest{}
	_ sql.Scanner    = (*Digest)(nil)
	_ slog.LogValuer = Digest{}
)

// Scan implements the sql.Scanner interface.
func (d *Digest) Scan(src any) error {
	if d.IsValid() {
		return errors.New("model.Digest: illegal Scan on valid Digest")
	}
	switch v := src.(type) {
	case string:
		*d = ParseDigest(v)
		return nil
	case []byte:
		*d = ParseDigest(string(v))
		return nil
	}
	return fmt.Errorf("model.Digest: invalid Scan source %T", src)
}

// Value implements the driver.Valuer interface.
func (d Digest) Value() (driver.Value, error) {
	return d.String(), nil
}

// ParseDigest parses a string in the form of "<digest-type>-<digest>" into a
// Digest.
func ParseDigest(s string) Digest {
	typ, digest, ok := strings.Cut(s, "-")
	if ok && isValidDigestType(typ) && isValidHex(digest) {
		return Digest{s: s}
	}
	return Digest{}
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
