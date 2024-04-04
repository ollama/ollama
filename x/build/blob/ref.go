package blob

import (
	"cmp"
	"fmt"
	"iter"
	"slices"
	"strings"
)

type Kind int

// Levels of concreteness
const (
	Domain Kind = iota
	Namespace
	Name
	Tag
	Build
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
	return with(r, Domain, s)
}

// WithNamespace returns a copy of r with the provided namespace. If the
// provided namespace is empty, it returns the short, unqualified copy of r.
func (r Ref) WithNamespace(s string) Ref {
	return with(r, Namespace, s)
}

func (r Ref) WithTag(s string) Ref {
	return with(r, Tag, s)
}

// WithBuild returns a copy of r with the provided build. If the provided
// build is empty, it returns the short, unqualified copy of r.
func (r Ref) WithBuild(s string) Ref {
	return with(r, Build, s)
}

func with(r Ref, kind Kind, value string) Ref {
	if value != "" && !isValidPart(value) {
		return Ref{}
	}
	switch kind {
	case Domain:
		r.domain = value
	case Namespace:
		r.namespace = value
	case Name:
		r.name = value
	case Tag:
		r.tag = value
	case Build:
		r.build = value
	default:
		panic(fmt.Sprintf("invalid completeness: %d", kind))
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

func (r Ref) CompleteWithoutBuild() bool {
	return r.Valid() && !slices.Contains(r.Parts()[:Tag], "")
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
		Domain:    r.domain,
		Namespace: r.namespace,
		Name:      r.name,
		Tag:       r.tag,
		Build:     r.build,
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
	var r Ref
	for kind, part := range Parts(s) {
		switch kind {
		case Domain:
			r.domain = part
		case Namespace:
			r.namespace = part
		case Name:
			r.name = part
		case Tag:
			r.tag = part
		case Build:
			r.build = part
		}
	}
	if !r.Valid() {
		return Ref{}
	}
	return r
}

func Parts(s string) iter.Seq2[Kind, string] {
	return func(yield func(Kind, string) bool) {
		if len(s) > 128 {
			return
		}

		if strings.HasPrefix(s, "http://") {
			s = s[len("http://"):]
		}
		if strings.HasPrefix(s, "https://") {
			s = s[len("https://"):]
		}

		state, j := Build, len(s)
		for i := len(s) - 1; i >= 0; i-- {
			c := s[i]
			switch c {
			case '+':
				switch state {
				case Build:
					v := s[i+1 : j]
					if v == "" {
						return
					}
					v = strings.ToUpper(v)
					if !yield(Build, v) {
						return
					}
					state, j = Tag, i
				default:
					return
				}
			case ':':
				switch state {
				case Build, Tag:
					v := s[i+1 : j]
					if v == "" {
						return
					}
					if !yield(Tag, v) {
						return
					}
					state, j = Name, i
				default:
					return
				}
			case '/':
				switch state {
				case Name, Tag, Build:
					if !yield(Name, s[i+1:j]) {
						return
					}
					state, j = Namespace, i
				case Namespace:
					if !yield(Namespace, s[i+1:j]) {
						return
					}
					state, j = Domain, i
				default:
					return
				}
			}
		}

		// handle the first part based on final state
		switch state {
		case Domain:
			yield(Domain, s[:j])
		case Namespace:
			yield(Namespace, s[:j])
		default:
			yield(Name, s[:j])
		}
	}
}

// Complete is the same as ParseRef(s).Complete().
//
// Future versions may be faster than calling ParseRef(s).Complete(), so if
// need to know if a ref is complete and don't need the ref, use this
// function.
func Complete(s string) bool {
	// TODO(bmizerany): fast-path this with a quick scan withput
	// allocating strings
	return ParseRef(s).Complete()
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
