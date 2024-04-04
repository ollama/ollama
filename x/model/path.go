package model

import (
	"cmp"
	"iter"
	"slices"
	"strings"
)

const MaxPathLength = 255

type PathPart int

// Levels of concreteness
const (
	Invalid PathPart = iota
	Domain
	Namespace
	Name
	Tag
	Build
)

var kindNames = map[PathPart]string{
	Invalid:   "Invalid",
	Domain:    "Domain",
	Namespace: "Namespace",
	Name:      "Name",
	Tag:       "Tag",
	Build:     "Build",
}

// Path is an opaque reference to a model.
//
// It is comparable and can be used as a map key.
//
// Users or Path must check Valid before using it.
type Path struct {
	domain    string
	namespace string
	name      string
	tag       string
	build     string
}

// Format returns a string representation of the ref with the given
// concreteness. If a part is missing, it is replaced with a loud
// placeholder.
func (r Path) Full() string {
	r.domain = cmp.Or(r.domain, "!(MISSING DOMAIN)")
	r.namespace = cmp.Or(r.namespace, "!(MISSING NAMESPACE)")
	r.name = cmp.Or(r.name, "!(MISSING NAME)")
	r.tag = cmp.Or(r.tag, "!(MISSING TAG)")
	r.build = cmp.Or(r.build, "!(MISSING BUILD)")
	return r.String()
}

func (r Path) NameAndTag() string {
	r.domain = ""
	r.namespace = ""
	r.build = ""
	return r.String()
}

func (r Path) NameTagAndBuild() string {
	r.domain = ""
	r.namespace = ""
	return r.String()
}

// String returns the fully qualified ref string.
func (r Path) String() string {
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

// Complete reports whether the ref is fully qualified. That is it has a
// domain, namespace, name, tag, and build.
func (r Path) Complete() bool {
	return r.Valid() && !slices.Contains(r.Parts(), "")
}

// CompleteWithoutBuild reports whether the ref would be complete if it had a
// valid build.
func (r Path) CompleteWithoutBuild() bool {
	r.build = "x"
	return r.Valid() && r.Complete()
}

// Less returns true if r is less concrete than o; false otherwise.
func (r Path) Less(o Path) bool {
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
func (r Path) Parts() []string {
	return []string{
		r.domain,
		r.namespace,
		r.name,
		r.tag,
		r.build,
	}
}

func (r Path) Domain() string    { return r.namespace }
func (r Path) Namespace() string { return r.namespace }
func (r Path) Name() string      { return r.name }
func (r Path) Tag() string       { return r.tag }
func (r Path) Build() string     { return r.build }

// ParsePath parses a model path string into a Path.
//
// Examples of valid paths:
//
//	"example.com/mistral:7b+x"
//	"example.com/mistral:7b+Q4_0"
//	"mistral:7b+x"
//	"example.com/x/mistral:latest+Q4_0"
//	"example.com/x/mistral:latest"
//
// Examples of invalid paths:
//
//	"example.com/mistral:7b+"
//	"example.com/mistral:7b+Q4_0+"
//	"x/y/z/z:8n+I"
//	""
func ParsePath(s string) Path {
	var r Path
	for kind, part := range PathParts(s) {
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
			r.build = strings.ToUpper(part)
		case Invalid:
			return Path{}
		}
	}
	if !r.Valid() {
		return Path{}
	}
	return r
}

// Merge folds the domain, namespace, tag, and build of b into a if not set.
// The name is left untouched.
//
// Use this for merging a ref with a default ref.
func Merge(a, b Path) Path {
	return Path{
		// name is left untouched
		name: a.name,

		domain:    cmp.Or(a.domain, b.domain),
		namespace: cmp.Or(a.namespace, b.namespace),
		tag:       cmp.Or(a.tag, b.tag),
		build:     cmp.Or(a.build, b.build),
	}
}

// WithBuild returns a copy of r with the build set to the given string.
func (r Path) WithBuild(build string) Path {
	r.build = build
	return r
}

// Parts returns a sequence of the parts of a ref string from most specific
// to least specific.
//
// It normalizes the input string by removing "http://" and "https://" only.
// No other normalization is done.
func PathParts(s string) iter.Seq2[PathPart, string] {
	return func(yield func(PathPart, string) bool) {
		if strings.HasPrefix(s, "http://") {
			s = s[len("http://"):]
		}
		if strings.HasPrefix(s, "https://") {
			s = s[len("https://"):]
		}

		if len(s) > MaxPathLength || len(s) == 0 {
			return
		}

		yieldValid := func(kind PathPart, part string) bool {
			if !isValidPart(part) {
				yield(Invalid, "")
				return false
			}
			return yield(kind, part)
		}

		state, j := Build, len(s)
		for i := len(s) - 1; i >= 0; i-- {
			switch s[i] {
			case '+':
				switch state {
				case Build:
					if !yieldValid(Build, s[i+1:j]) {
						return
					}
					state, j = Tag, i
				default:
					yield(Invalid, "")
					return
				}
			case ':':
				switch state {
				case Build, Tag:
					if !yieldValid(Tag, s[i+1:j]) {
						return
					}
					state, j = Name, i
				default:
					yield(Invalid, "")
					return
				}
			case '/':
				switch state {
				case Name, Tag, Build:
					if !yieldValid(Name, s[i+1:j]) {
						return
					}
					state, j = Namespace, i
				case Namespace:
					if !yieldValid(Namespace, s[i+1:j]) {
						return
					}
					state, j = Domain, i
				default:
					yield(Invalid, "")
					return
				}
			default:
				if !isValidPart(s[i : i+1]) {
					yield(Invalid, "")
					return
				}
			}
		}

		if state <= Namespace {
			yieldValid(state, s[:j])
		} else {
			yieldValid(Name, s[:j])
		}
	}
}

// Valid returns true if the ref has a valid name. To know if a ref is
// "complete", use Complete.
func (r Path) Valid() bool {
	// Parts ensures we only have valid parts, so no need to validate
	// them here, only check if we have a name or not.
	return r.name != ""
}

// isValidPart returns true if given part is valid ascii [a-zA-Z0-9_\.-]
func isValidPart(s string) bool {
	if s == "" {
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
