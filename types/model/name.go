// Package model contains types and utilities for parsing, validating, and
// working with model names and digests.
package model

import (
	"cmp"
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

const (
	defaultHost      = "registry.ollama.ai"
	defaultNamespace = "library"
	defaultTag       = "latest"
)

// DefaultName returns a name with the default values for the host, namespace,
// and tag parts. The model and digest parts are empty.
//
//   - The default host is ("registry.ollama.ai")
//   - The default namespace is ("library")
//   - The default tag is ("latest")
func DefaultName() Name {
	return Name{
		Host:      defaultHost,
		Namespace: defaultNamespace,
		Tag:       defaultTag,
	}
}

type partKind int

const (
	kindHost partKind = iota
	kindNamespace
	kindKind
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
	case kindKind:
		return "kind"
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
type Name struct {
	Host      string
	Namespace string
	Kind      string // Optional: "skill", "agent", or empty for models
	Model     string
	Tag       string
}

// ParseName parses and assembles a Name from a name string. The
// format of a valid name string is:
//
//	  s:
//		  { host } "/" { namespace } "/" { kind } "/" { model } ":" { tag }
//		  { host } "/" { namespace } "/" { model } ":" { tag }
//		  { namespace } "/" { kind } "/" { model } ":" { tag }
//		  { namespace } "/" { model } ":" { tag }
//		  { model } ":" { tag }
//		  { model }
//	  host:
//	      pattern: { alphanum | "_" } { alphanum | "-" | "_" | "." | ":" }*
//	      length:  [1, 350]
//	  namespace:
//	      pattern: { alphanum | "_" } { alphanum | "-" | "_" }*
//	      length:  [1, 80]
//	  kind:
//	      pattern: "skill" | "agent" | "" (empty for models)
//	      length:  [0, 80]
//	  model:
//	      pattern: { alphanum | "_" } { alphanum | "-" | "_" | "." }*
//	      length:  [1, 80]
//	  tag:
//	      pattern: { alphanum | "_" } { alphanum | "-" | "_" | "." }*
//	      length:  [1, 80]
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

// ValidKinds are the allowed values for the Kind field
var ValidKinds = map[string]bool{
	"skill": true,
	"agent": true,
	"mcp":   true,
}

// ParseNameBare parses s as a name string and returns a Name. No merge with
// [DefaultName] is performed.
func ParseNameBare(s string) Name {
	var n Name
	var promised bool

	// "/" is an illegal tag character, so we can use it to split the host
	if strings.LastIndex(s, ":") > strings.LastIndex(s, "/") {
		s, n.Tag, _ = cutPromised(s, ":")
	}

	s, n.Model, promised = cutPromised(s, "/")
	if !promised {
		n.Model = s
		return n
	}

	s, n.Kind, promised = cutPromised(s, "/")
	if !promised {
		// Only 2 parts: namespace/model - what we parsed as Kind is actually Namespace
		n.Namespace = n.Kind
		n.Kind = ""
		return n
	}

	// Check if what we parsed as Kind is actually a valid kind value
	if !ValidKinds[n.Kind] {
		// Not a valid kind - this is the old 3-part format: host/namespace/model
		// Shift: Kind -> Namespace, s -> Host
		n.Namespace = n.Kind
		n.Kind = ""

		scheme, host, ok := strings.Cut(s, "://")
		if !ok {
			host = scheme
		}
		n.Host = host
		return n
	}

	// Valid kind found - continue parsing for namespace and optional host
	s, n.Namespace, promised = cutPromised(s, "/")
	if !promised {
		n.Namespace = s
		return n
	}

	scheme, host, ok := strings.Cut(s, "://")
	if !ok {
		host = scheme
	}
	n.Host = host

	return n
}

// ParseNameFromFilepath parses a 4 or 5-part filepath as a Name. The parts are
// expected to be in the form:
//
// { host } "/" { namespace } "/" { model } "/" { tag }
// { host } "/" { namespace } "/" { kind } "/" { model } "/" { tag }
func ParseNameFromFilepath(s string) (n Name) {
	parts := strings.Split(s, string(filepath.Separator))

	switch len(parts) {
	case 4:
		// Old format: host/namespace/model/tag
		n.Host = parts[0]
		n.Namespace = parts[1]
		n.Model = parts[2]
		n.Tag = parts[3]
	case 5:
		// New format: host/namespace/kind/model/tag
		n.Host = parts[0]
		n.Namespace = parts[1]
		n.Kind = parts[2]
		n.Model = parts[3]
		n.Tag = parts[4]
	default:
		return Name{}
	}

	if !n.IsFullyQualified() {
		return Name{}
	}

	return n
}

// Merge merges the host, namespace, kind, and tag parts of the two names,
// preferring the non-empty parts of a.
func Merge(a, b Name) Name {
	a.Host = cmp.Or(a.Host, b.Host)
	a.Namespace = cmp.Or(a.Namespace, b.Namespace)
	a.Kind = cmp.Or(a.Kind, b.Kind)
	a.Tag = cmp.Or(a.Tag, b.Tag)
	return a
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
	if n.Kind != "" {
		b.WriteString(n.Kind)
		b.WriteByte('/')
	}
	b.WriteString(n.Model)
	if n.Tag != "" {
		b.WriteByte(':')
		b.WriteString(n.Tag)
	}
	return b.String()
}

// DisplayShortest returns a short string version of the name.
func (n Name) DisplayShortest() string {
	var sb strings.Builder

	if !strings.EqualFold(n.Host, defaultHost) {
		sb.WriteString(n.Host)
		sb.WriteByte('/')
		sb.WriteString(n.Namespace)
		sb.WriteByte('/')
	} else if !strings.EqualFold(n.Namespace, defaultNamespace) {
		sb.WriteString(n.Namespace)
		sb.WriteByte('/')
	}

	// include kind if present
	if n.Kind != "" {
		sb.WriteString(n.Kind)
		sb.WriteByte('/')
	}

	// always include model and tag
	sb.WriteString(n.Model)
	sb.WriteString(":")
	sb.WriteString(n.Tag)
	return sb.String()
}

// IsValidNamespace reports whether the provided string is a valid
// namespace.
func IsValidNamespace(s string) bool {
	return isValidPart(kindNamespace, s)
}

// IsValid reports whether all parts of the name are present and valid. The
// digest is a special case, and is checked for validity only if present.
//
// Note: The digest check has been removed as is planned to be added back in
// at a later time.
func (n Name) IsValid() bool {
	return n.IsFullyQualified()
}

// IsFullyQualified returns true if all parts of the name are present and
// valid without the digest. Kind is optional and only validated if non-empty.
func (n Name) IsFullyQualified() bool {
	if !isValidPart(kindHost, n.Host) {
		return false
	}
	if !isValidPart(kindNamespace, n.Namespace) {
		return false
	}
	// Kind is optional - only validate if present
	if n.Kind != "" && !isValidPart(kindKind, n.Kind) {
		return false
	}
	if !isValidPart(kindModel, n.Model) {
		return false
	}
	if !isValidPart(kindTag, n.Tag) {
		return false
	}
	return true
}

// Filepath returns a canonical filepath that represents the name with each part from
// host to tag as a directory in the form:
//
//	{host}/{namespace}/{model}/{tag}
//	{host}/{namespace}/{kind}/{model}/{tag}
//
// It uses the system's filepath separator and ensures the path is clean.
//
// It panics if the name is not fully qualified. Use [Name.IsFullyQualified]
// to check if the name is fully qualified.
func (n Name) Filepath() string {
	if !n.IsFullyQualified() {
		panic("illegal attempt to get filepath of invalid name")
	}
	if n.Kind != "" {
		return filepath.Join(
			n.Host,
			n.Namespace,
			n.Kind,
			n.Model,
			n.Tag,
		)
	}
	return filepath.Join(
		n.Host,
		n.Namespace,
		n.Model,
		n.Tag,
	)
}

// LogValue returns a slog.Value that represents the name as a string.
func (n Name) LogValue() slog.Value {
	return slog.StringValue(n.String())
}

func (n Name) EqualFold(o Name) bool {
	return strings.EqualFold(n.Host, o.Host) &&
		strings.EqualFold(n.Namespace, o.Namespace) &&
		strings.EqualFold(n.Kind, o.Kind) &&
		strings.EqualFold(n.Model, o.Model) &&
		strings.EqualFold(n.Tag, o.Tag)
}

func isValidLen(kind partKind, s string) bool {
	switch kind {
	case kindHost:
		return len(s) >= 1 && len(s) <= 350
	case kindTag:
		return len(s) >= 1 && len(s) <= 80
	default:
		return len(s) >= 1 && len(s) <= 80
	}
}

func isValidPart(kind partKind, s string) bool {
	// Kind must be one of the valid values
	if kind == kindKind {
		return ValidKinds[s]
	}

	if !isValidLen(kind, s) {
		return false
	}
	for i := range s {
		if i == 0 {
			if !isAlphanumericOrUnderscore(s[i]) {
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
			if kind != kindHost && kind != kindDigest {
				return false
			}
		default:
			if !isAlphanumericOrUnderscore(s[i]) {
				return false
			}
		}
	}
	return true
}

func isAlphanumericOrUnderscore(c byte) bool {
	return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_'
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
