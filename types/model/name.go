// Package model contains types and utilities for parsing, validating, and
// working with model names and digests.
package model

import (
	"cmp"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"path/filepath"
	"strings"
)

// Errors
var (
	// ErrUnqualifiedName represents an error where a name is not fully
	// qualified. It is not used directly in this package, but is here
	// to avoid other packages inventing their own error type.
	// Additionally, it can be conveniently used via [Unqualified].
	ErrUnqualifiedName = errors.New("unqualified name")
)

// Unqualified is a helper function that returns an error with
// ErrUnqualifiedName as the cause and the name as the message.
func Unqualified(n Name) error {
	return fmt.Errorf("%w: %s", ErrUnqualifiedName, n)
}

// MissingPart is used to indicate any part of a name that was "promised" by
// the presence of a separator, but is missing.
//
// The value was chosen because it is deemed unlikely to be set by a user,
// not a valid part name valid when checked by [Name.IsValid], and easy to
// spot in logs.
const MissingPart = "!MISSING!"

// DefaultName returns a name with the default values for the host, namespace,
// and tag parts. The model and digest parts are empty.
//
//   - The default host is ("registry.ollama.ai")
//   - The default namespace is ("library")
//   - The default tag is ("latest")
func DefaultName() Name {
	return Name{
		Host:      "registry.ollama.ai",
		Namespace: "library",
		Tag:       "latest",
	}
}

type partKind int

const (
	kindHost partKind = iota
	kindNamespace
	kindModel
	kindTag
	kindDigest
)

func (k partKind) String() string {
	switch k {
	case kindHost:
		return "host"
	case kindNamespace:
		return "namespace"
	case kindModel:
		return "model"
	case kindTag:
		return "tag"
	case kindDigest:
		return "digest"
	default:
		return "unknown"
	}
}

// Name is a structured representation of a model name string, as defined by
// [ParseNameNoDefaults].
//
// It is not guaranteed to be valid. Use [Name.IsValid] to check if the name
// is valid.
//
// It is not directly comparable with other Names. Use [Name.Equal] and
// [Name.MapHash] for determining equality and using as a map key.
type Name struct {
	Host      string
	Namespace string
	Model     string
	Tag       string
	RawDigest string
}

// ParseName parses and assembles a Name from a name string. The
// format of a valid name string is:
//
//	  s:
//		  { host } "/" { namespace } "/" { model } ":" { tag } "@" { digest }
//		  { host } "/" { namespace } "/" { model } ":" { tag }
//		  { host } "/" { namespace } "/" { model } "@" { digest }
//		  { host } "/" { namespace } "/" { model }
//		  { namespace } "/" { model } ":" { tag } "@" { digest }
//		  { namespace } "/" { model } ":" { tag }
//		  { namespace } "/" { model } "@" { digest }
//		  { namespace } "/" { model }
//		  { model } ":" { tag } "@" { digest }
//		  { model } ":" { tag }
//		  { model } "@" { digest }
//		  { model }
//		  "@" { digest }
//	  host:
//	      pattern: alphanum { alphanum | "-" | "_" | "." | ":" }*
//	      length:  [1, 350]
//	  namespace:
//	      pattern: alphanum { alphanum | "-" | "_" }*
//	      length:  [2, 80]
//	  model:
//	      pattern: alphanum { alphanum | "-" | "_" | "." }*
//	      length:  [2, 80]
//	  tag:
//	      pattern: alphanum { alphanum | "-" | "_" | "." }*
//	      length:  [1, 80]
//	  digest:
//	      pattern: alphanum { alphanum | "-" | ":" }*
//	      length:  [2, 80]
//
// Most users should use [ParseName] instead, unless need to support
// different defaults than DefaultName.
//
// The name returned is not guaranteed to be valid. If it is not valid, the
// field values are left in an undefined state. Use [Name.IsValid] to check
// if the name is valid.
func ParseName(s string) Name {
	return Merge(ParseNameBare(s), DefaultName())
}

// ParseNameBare parses s as a name string and returns a Name. No merge with
// [DefaultName] is performed.
func ParseNameBare(s string) Name {
	var n Name
	var promised bool

	s, n.RawDigest, promised = cutLast(s, "@")
	if promised && n.RawDigest == "" {
		n.RawDigest = MissingPart
	}

	s, n.Tag, _ = cutPromised(s, ":")
	s, n.Model, promised = cutPromised(s, "/")
	if !promised {
		n.Model = s
		return n
	}
	s, n.Namespace, promised = cutPromised(s, "/")
	if !promised {
		n.Namespace = s
		return n
	}
	n.Host = s

	return n
}

// Merge merges the host, namespace, and tag parts of the two names,
// preferring the non-empty parts of a.
func Merge(a, b Name) Name {
	a.Host = cmp.Or(a.Host, b.Host)
	a.Namespace = cmp.Or(a.Namespace, b.Namespace)
	a.Tag = cmp.Or(a.Tag, b.Tag)
	return a
}

// Digest returns the result of [ParseDigest] with the RawDigest field.
func (n Name) Digest() Digest {
	return ParseDigest(n.RawDigest)
}

// String returns the name string, in the format that [ParseNameNoDefaults]
// accepts as valid, if [Name.IsValid] reports true; otherwise the empty
// string is returned.
func (n Name) String() string {
	var b strings.Builder
	if n.Host != "" {
		b.WriteString(n.Host)
		b.WriteByte('/')
	}
	if n.Namespace != "" {
		b.WriteString(n.Namespace)
		b.WriteByte('/')
	}
	b.WriteString(n.Model)
	if n.Tag != "" {
		b.WriteByte(':')
		b.WriteString(n.Tag)
	}
	if n.RawDigest != "" {
		b.WriteByte('@')
		b.WriteString(n.RawDigest)
	}
	return b.String()
}

// IsValid reports whether all parts of the name are present and valid. The
// digest is a special case, and is checked for validity only if present.
func (n Name) IsValid() bool {
	if n.RawDigest != "" && !ParseDigest(n.RawDigest).IsValid() {
		return false
	}
	return n.IsFullyQualified()
}

// IsFullyQualified returns true if all parts of the name are present and
// valid without the digest.
func (n Name) IsFullyQualified() bool {
	var parts = []string{
		n.Host,
		n.Namespace,
		n.Model,
		n.Tag,
	}
	for i, part := range parts {
		if !isValidPart(partKind(i), part) {
			return false
		}
	}
	return true
}

// Filepath returns a canonical filepath that represents the name with each part from
// host to tag as a directory in the form:
//
//	{host}/{namespace}/{model}/{tag}
//
// It uses the system's filepath separator and ensures the path is clean.
//
// It panics if the name is not fully qualified. Use [Name.IsFullyQualified]
// to check if the name is fully qualified.
func (n Name) Filepath() string {
	if !n.IsFullyQualified() {
		panic("illegal attempt to get filepath of invalid name")
	}
	return filepath.Join(
		strings.ToLower(n.Host),
		strings.ToLower(n.Namespace),
		strings.ToLower(n.Model),
		strings.ToLower(n.Tag),
	)
}

// LogValue returns a slog.Value that represents the name as a string.
func (n Name) LogValue() slog.Value {
	return slog.StringValue(n.String())
}

func isValidLen(kind partKind, s string) bool {
	switch kind {
	case kindHost:
		return len(s) >= 1 && len(s) <= 350
	case kindTag:
		return len(s) >= 1 && len(s) <= 80
	default:
		return len(s) >= 2 && len(s) <= 80
	}
}

func isValidPart(kind partKind, s string) bool {
	if !isValidLen(kind, s) {
		return false
	}
	for i := range s {
		if i == 0 {
			if !isAlphanumeric(s[i]) {
				return false
			}
			continue
		}
		switch s[i] {
		case '_', '-':
		case '.':
			if kind == kindNamespace {
				return false
			}
		case ':':
			if kind != kindHost {
				return false
			}
		default:
			if !isAlphanumeric(s[i]) {
				return false
			}
		}
	}
	return true
}

func isAlphanumeric(c byte) bool {
	return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9'
}

func cutLast(s, sep string) (before, after string, ok bool) {
	i := strings.LastIndex(s, sep)
	if i >= 0 {
		return s[:i], s[i+len(sep):], true
	}
	return s, "", false
}

// cutPromised cuts the last part of s at the last occurrence of sep. If sep is
// found, the part before and after sep are returned as-is unless empty, in
// which case they are returned as MissingPart, which will cause
// [Name.IsValid] to return false.
func cutPromised(s, sep string) (before, after string, ok bool) {
	before, after, ok = cutLast(s, sep)
	if !ok {
		return before, after, false
	}
	return cmp.Or(before, MissingPart), cmp.Or(after, MissingPart), true
}

type DigestType int

const (
	DigestTypeInvalid DigestType = iota
	DigestTypeSHA256
)

func (t DigestType) String() string {
	if t == DigestTypeSHA256 {
		return "sha256"
	}
	return "unknown"
}

// Digest represents a type and hash of a digest. It is comparable and can
// be used as a map key.
type Digest struct {
	Type DigestType
	Hash [32]byte
}

// ParseDigest parses a digest string into a Digest struct. It accepts both
// the forms:
//
//	sha256:deadbeef
//	sha256-deadbeef
//
// The hash part must be exactly 64 characters long.
//
// The form "type:hash" does not round trip through [Digest.String].
func ParseDigest(s string) Digest {
	typ, hash, ok := cutLast(s, ":")
	if !ok {
		typ, hash, ok = cutLast(s, "-")
		if !ok {
			return Digest{}
		}
	}
	if typ != "sha256" {
		return Digest{}
	}
	var d Digest
	n, err := hex.Decode(d.Hash[:], []byte(hash))
	if err != nil || n != 32 {
		return Digest{}
	}
	return Digest{Type: DigestTypeSHA256, Hash: d.Hash}
}

// IsValid returns true if the digest has a valid Type and Hash.
func (d Digest) IsValid() bool {
	if d.Type != DigestTypeSHA256 {
		return false
	}
	return d.Hash != [32]byte{}
}

// String returns the digest as a string in the form "type-hash". The hash
// is encoded as a hex string.
func (d Digest) String() string {
	var b strings.Builder
	b.WriteString(d.Type.String())
	b.WriteByte('-')
	b.WriteString(hex.EncodeToString(d.Hash[:]))
	return b.String()
}

// LogValue returns a slog.Value that represents the digest as a string.
func (d Digest) LogValue() slog.Value {
	return slog.StringValue(d.String())
}
