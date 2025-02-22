package names

import (
	"cmp"
	"fmt"
	"strings"

	"github.com/ollama/ollama/server/internal/internal/stringsx"
)

const MaxNameLength = 50 + 1 + 50 + 1 + 50 // <namespace>/<model>:<tag>

type Name struct {
	// Make incomparable to enfoce use of Compare / Equal for
	// case-insensitive comparisons.
	_ [0]func()

	h string
	n string
	m string
	t string
}

// Parse parses and assembles a Name from a name string. The
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
//	      pattern: { alphanum | "_" } { alphanum | "_" | "-" | "." | ":" }*
//	      length:  [1, 350]
//	  namespace:
//	      pattern: { alphanum | "_" } { alphanum | "-" | "_" }*
//	      length:  [1, 80]
//	  model:
//	      pattern: { alphanum | "_" } { alphanum | "-" | "_" | "." }*
//	      length:  [1, 80]
//	  tag:
//	      pattern: { alphanum | "_" } { alphanum | "-" | "_" | "." }*
//	      length:  [1, 80]
//	  digest:
//	      pattern: { alphanum | "_" } { alphanum | "-" | ":" }*
//	      length:  [1, 80]
//
// The name returned is not guaranteed to be valid. If it is not valid, the
// field values are left in an undefined state. Use [Name.IsValid] to check
// if the name is valid.
func Parse(s string) Name {
	if len(s) > MaxNameLength {
		return Name{}
	}

	var n Name
	var tail string
	var c byte
	for {
		s, tail, c = cutLastAny(s, "/:")
		switch c {
		case ':':
			n.t = tail
			continue // look for model
		case '/':
			n.h, n.n, _ = cutLastAny(s, "/")
			n.m = tail
			return n
		case 0:
			n.m = tail
			return n
		}
	}
}

// ParseExtended parses and returns any scheme, Name, and digest from from s in
// the the form [scheme://][name][@digest]. All parts are optional.
//
// If the scheme is present, it must be followed by "://". The digest is
// prefixed by "@" and comes after the name. The name is parsed using [Parse].
//
// The scheme and digest are stripped before the name is parsed by [Parse].
//
// For convience, the scheme is never empty. If the scheme is not present, the
// returned scheme is "https".
//
// Examples:
//
//	http://ollama.com/bmizerany/smol:latest@digest
//	https://ollama.com/bmizerany/smol:latest
//	ollama.com/bmizerany/smol:latest@digest // returns "https" scheme.
func ParseExtended(s string) (scheme string, _ Name, digest string) {
	i := strings.Index(s, "://")
	if i >= 0 {
		scheme = s[:i]
		s = s[i+3:]
	}
	i = strings.LastIndex(s, "@")
	if i >= 0 {
		digest = s[i+1:]
		s = s[:i]
	}
	return scheme, Parse(s), digest
}

func FormatExtended(scheme string, n Name, digest string) string {
	var b strings.Builder
	if scheme != "" {
		b.WriteString(scheme)
		b.WriteString("://")
	}
	b.WriteString(n.String())
	if digest != "" {
		b.WriteByte('@')
		b.WriteString(digest)
	}
	return b.String()
}

// Merge merges two names into a single name. Non-empty host, namespace, and
// tag parts of a take precedence over fields in b. The model field is left as
// is.
//
// The returned name is not guaranteed to be valid. Use [Name.IsValid] to check
// if the name is valid.
func Merge(a, b Name) Name {
	a.h = cmp.Or(a.h, b.h)
	a.n = cmp.Or(a.n, b.n)
	a.t = cmp.Or(a.t, b.t)
	return a
}

// IsValid returns true if the name is valid.
func (n Name) IsValid() bool {
	if n.h != "" && !isValidHost(n.h) {
		return false
	}
	if n.n != "" && !isValidNamespace(n.n) {
		return false
	}
	if n.m != "" && !isValidModel(n.m) {
		return false
	}
	if n.t != "" && !isValidTag(n.t) {
		return false
	}
	return true
}

func (n Name) IsFullyQualified() bool {
	return n.IsValid() && n.h != "" && n.n != "" && n.m != "" && n.t != ""
}

func isValidHost(_ string) bool {
	return true // TODO: implement
}

func isValidNamespace(_ string) bool {
	return true // TODO: implement
}

func isValidModel(_ string) bool {
	return true // TODO: implement
}

func isValidTag(_ string) bool {
	return true // TODO: implement
}

func (n Name) Host() string      { return n.h }
func (n Name) Namespace() string { return n.n }
func (n Name) Model() string     { return n.m }
func (n Name) Tag() string       { return n.t }

// Compare compares n and o case-insensitively. It returns 0 if n and o are
// equal, -1 if n sorts before o, and 1 if n sorts after o.
func (n Name) Compare(o Name) int {
	return cmp.Or(
		stringsx.CompareFold(n.h, o.h),
		stringsx.CompareFold(n.n, o.n),
		stringsx.CompareFold(n.m, o.m),
		stringsx.CompareFold(n.t, o.t),
	)
}

// String returns the fully qualified name in the format
// <namespace>/<model>:<tag>.
func (n Name) String() string {
	var b strings.Builder
	if n.h != "" {
		b.WriteString(n.h)
		b.WriteByte('/')
	}
	if n.n != "" {
		b.WriteString(n.n)
		b.WriteByte('/')
	}
	b.WriteString(n.m)
	if n.t != "" {
		b.WriteByte(':')
		b.WriteString(n.t)
	}
	return b.String()
}

func (n Name) GoString() string {
	return fmt.Sprintf("<Name %q %q %q %q>", n.h, n.n, n.m, n.t)
}

// cutLastAny is like strings.Cut but scans in reverse for the last character
// in chars. If no character is found, before is the empty string and after is
// s. The returned sep is the byte value of the character in chars if one was
// found; otherwise it is 0.
func cutLastAny(s, chars string) (before, after string, sep byte) {
	i := strings.LastIndexAny(s, chars)
	if i >= 0 {
		return s[:i], s[i+1:], s[i]
	}
	return "", s, 0
}
