package blob

import (
	"cmp"
	"fmt"
	"slices"
	"strings"
)

// Levels of concreteness
const (
	domain = iota
	namespace
	name
	tag
	build
)

// Ref is an opaque reference to a blob.
//
// It is comparable and can be used as a map key.
//
// Users or Ref must check Valid before using it.
type Ref struct {
	domain    string
	namespace string
	name      string
	tag       string
	build     string
}

// WithDomain returns a copy of r with the provided domain. If the provided
// domain is empty, it returns the short, unqualified copy of r.
func (r Ref) WithDomain(s string) Ref {
	return with(r, domain, s)
}

// WithNamespace returns a copy of r with the provided namespace. If the
// provided namespace is empty, it returns the short, unqualified copy of r.
func (r Ref) WithNamespace(s string) Ref {
	return with(r, namespace, s)
}

func (r Ref) WithTag(s string) Ref {
	return with(r, tag, s)
}

// WithBuild returns a copy of r with the provided build. If the provided
// build is empty, it returns the short, unqualified copy of r.
func (r Ref) WithBuild(s string) Ref {
	return with(r, build, s)
}

func with(r Ref, part int, value string) Ref {
	if value != "" && !isValidPart(value) {
		return Ref{}
	}
	switch part {
	case domain:
		r.domain = value
	case namespace:
		r.namespace = value
	case name:
		r.name = value
	case tag:
		r.tag = value
	case build:
		r.build = value
	default:
		panic(fmt.Sprintf("invalid completeness: %d", part))
	}
	return r
}

// Format returns a string representation of the ref with the given
// concreteness. If a part is missing, it is replaced with a loud
// placeholder.
func (r Ref) Full() string {
	r.domain = cmp.Or(r.domain, "!(MISSING DOMAIN)")
	r.namespace = cmp.Or(r.namespace, "!(MISSING NAMESPACE)")
	r.name = cmp.Or(r.name, "!(MISSING NAME)")
	r.tag = cmp.Or(r.tag, "!(MISSING TAG)")
	r.build = cmp.Or(r.build, "!(MISSING BUILD)")
	return r.String()
}

func (r Ref) NameAndTag() string {
	r.domain = ""
	r.namespace = ""
	r.build = ""
	return r.String()
}

func (r Ref) NameTagAndBuild() string {
	r.domain = ""
	r.namespace = ""
	return r.String()
}

// String returns the fully qualified ref string.
func (r Ref) String() string {
	var b strings.Builder
	if r.domain != "" {
		b.WriteString(r.domain)
		b.WriteString("/")
	}
	if r.namespace != "" {
		b.WriteString(r.namespace)
		b.WriteString("/")
	}
	b.WriteString(r.name)
	if r.tag != "" {
		b.WriteString(":")
		b.WriteString(r.tag)
	}
	if r.build != "" {
		b.WriteString("+")
		b.WriteString(r.build)
	}
	return b.String()
}

// Complete returns true if the ref is valid and has no empty parts.
func (r Ref) Complete() bool {
	return r.Valid() && !slices.Contains(r.Parts(), "")
}

// Less returns true if r is less concrete than o; false otherwise.
func (r Ref) Less(o Ref) bool {
	rp := r.Parts()
	op := o.Parts()
	for i := range rp {
		if rp[i] < op[i] {
			return true
		}
	}
	return false
}

// Parts returns the parts of the ref in order of concreteness.
//
// The length of the returned slice is always 5.
func (r Ref) Parts() []string {
	return []string{
		domain:    r.domain,
		namespace: r.namespace,
		name:      r.name,
		tag:       r.tag,
		build:     r.build,
	}
}

func (r Ref) Domain() string    { return r.namespace }
func (r Ref) Namespace() string { return r.namespace }
func (r Ref) Name() string      { return r.name }
func (r Ref) Tag() string       { return r.tag }
func (r Ref) Build() string     { return r.build }

// ParseRef parses a ref string into a Ref. A ref string is a name, an
// optional tag, and an optional build, separated by colons and pluses.
//
// The name must be valid ascii [a-zA-Z0-9_].
// The tag must be valid ascii [a-zA-Z0-9_].
// The build must be valid ascii [a-zA-Z0-9_].
//
// It returns then zero value if the ref is invalid.
//
//	// Valid Examples:
//	ParseRef("mistral:latest") returns ("mistral", "latest", "")
//	ParseRef("mistral") returns ("mistral", "", "")
//	ParseRef("mistral:30B") returns ("mistral", "30B", "")
//	ParseRef("mistral:7b") returns ("mistral", "7b", "")
//	ParseRef("mistral:7b+Q4_0") returns ("mistral", "7b", "Q4_0")
//	ParseRef("mistral+KQED") returns ("mistral", "latest", "KQED")
//	ParseRef(".x.:7b+Q4_0:latest") returns (".x.", "7b", "Q4_0")
//	ParseRef("-grok-f.oo:7b+Q4_0") returns ("-grok-f.oo", "7b", "Q4_0")
//
//	// Invalid Examples:
//	ParseRef("m stral") returns ("", "", "") // zero
//	ParseRef("... 129 chars ...") returns ("", "", "") // zero
func ParseRef(s string) Ref {
	if len(s) > 128 {
		return Ref{}
	}

	if strings.HasPrefix(s, "http://") {
		s = s[len("http://"):]
	}
	if strings.HasPrefix(s, "https://") {
		s = s[len("https://"):]
	}

	var r Ref

	state, j := build, len(s)
	for i := len(s) - 1; i >= 0; i-- {
		c := s[i]
		switch c {
		case '+':
			switch state {
			case build:
				r.build = s[i+1 : j]
				r.build = strings.ToUpper(r.build)
				state, j = tag, i
			default:
				return Ref{}
			}
		case ':':
			switch state {
			case build, tag:
				r.tag = s[i+1 : j]
				state, j = name, i
			default:
				return Ref{}
			}
		case '/':
			switch state {
			case name, tag, build:
				r.name = s[i+1 : j]
				state, j = namespace, i
			case namespace:
				r.namespace = s[i+1 : j]
				state, j = domain, i
			default:
				return Ref{}
			}
		}
	}

	// handle the first part based on final state
	switch state {
	case domain:
		r.domain = s[:j]
	case namespace:
		r.namespace = s[:j]
	default:
		r.name = s[:j]
	}

	if !r.Valid() {
		return Ref{}
	}
	return r
}

func (r Ref) Valid() bool {
	// Name is required
	if !isValidPart(r.name) {
		return false
	}

	// Optional parts must be valid if present
	if r.domain != "" && !isValidPart(r.domain) {
		return false
	}
	if r.namespace != "" && !isValidPart(r.namespace) {
		return false
	}
	if r.tag != "" && !isValidPart(r.tag) {
		return false
	}
	if r.build != "" && !isValidPart(r.build) {
		return false
	}
	return true
}

// isValidPart returns true if given part is valid ascii [a-zA-Z0-9_\.-]
func isValidPart(s string) bool {
	if len(s) == 0 {
		return false
	}
	for _, c := range []byte(s) {
		if c == '.' || c == '-' {
			return true
		}
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' {
			continue
		} else {
			return false

		}
	}
	return true
}
