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

// Digest is an opaque reference to a model digest. It holds the digest type
// and the digest itself.
//
// It is comparable with other Digests and can be used as a map key.
type Digest struct {
	s string
}

func (d Digest) Type() string {
	typ, _, _ := strings.Cut(d.s, "-")
	return typ
}

func (d Digest) IsValid() bool  { return d.s != "" }
func (d Digest) String() string { return d.s }

func (d Digest) MarshalText() ([]byte, error) {
	return []byte(d.String()), nil
}

func (d *Digest) UnmarshalText(text []byte) error {
	if d.IsValid() {
		return errors.New("model.Digest: illegal UnmarshalText on valid Digest")
	}
	*d = ParseDigest(string(text))
	return nil
}

func (d Digest) LogValue() slog.Value {
	return slog.StringValue(d.String())
}

var (
	_ driver.Valuer = Digest{}
	_ sql.Scanner   = (*Digest)(nil)
)

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

// isValidDigest returns true if the given string in the form of
// "<digest-type>-<digest>", and <digest-type> is in the form of [a-z0-9]+
// and <digest> is a valid hex string.
//
// It does not check if the digest is a valid hash for the given digest
// type, or restrict the digest type to a known set of types. This is left
// up to ueers of this package.
func isValidDigest(s string) bool {
	typ, digest, ok := strings.Cut(s, "-")
	res := ok && isValidDigestType(typ) && isValidHex(digest)
	fmt.Printf("DEBUG: %q: typ: %s, digest: %s, ok: %v res: %v\n", s, typ, digest, ok, res)
	return res
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
