package model

import (
	"cmp"
	"iter"
	"slices"
	"strings"
)

const MaxNameLength = 255

type NamePart int

// Levels of concreteness
const (
	Invalid NamePart = iota
	Registry
	Namespace
	Nick
	Tag
	Build
)

var kindNames = map[NamePart]string{
	Invalid:   "Invalid",
	Registry:  "Domain",
	Namespace: "Namespace",
	Nick:      "Name",
	Tag:       "Tag",
	Build:     "Build",
}

// Name is an opaque reference to a model.
//
// It is comparable and can be used as a map key.
//
// Users or Name must check Valid before using it.
type Name struct {
	domain    string
	namespace string
	nick      string
	tag       string
	build     string
}

// Format returns a string representation of the ref with the given
// concreteness. If a part is missing, it is replaced with a loud
// placeholder.
func (r Name) Full() string {
	r.domain = cmp.Or(r.domain, "!(MISSING DOMAIN)")
	r.namespace = cmp.Or(r.namespace, "!(MISSING NAMESPACE)")
	r.nick = cmp.Or(r.nick, "!(MISSING NAME)")
	r.tag = cmp.Or(r.tag, "!(MISSING TAG)")
	r.build = cmp.Or(r.build, "!(MISSING BUILD)")
	return r.String()
}

func (r Name) NickAndTag() string {
	r.domain = ""
	r.namespace = ""
	r.build = ""
	return r.String()
}

func (r Name) NickTagAndBuild() string {
	r.domain = ""
	r.namespace = ""
	return r.String()
}

// String returns the fully qualified ref string.
func (r Name) String() string {
	var b strings.Builder
	if r.domain != "" {
		b.WriteString(r.domain)
		b.WriteString("/")
	}
	if r.namespace != "" {
		b.WriteString(r.namespace)
		b.WriteString("/")
	}
	b.WriteString(r.nick)
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
func (r Name) Complete() bool {
	return r.Valid() && !slices.Contains(r.Parts(), "")
}

// CompleteWithoutBuild reports whether the ref would be complete if it had a
// valid build.
func (r Name) CompleteWithoutBuild() bool {
	r.build = "x"
	return r.Valid() && r.Complete()
}

// Less returns true if r is less concrete than o; false otherwise.
func (r Name) Less(o Name) bool {
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
func (r Name) Parts() []string {
	return []string{
		r.domain,
		r.namespace,
		r.nick,
		r.tag,
		r.build,
	}
}

func (r Name) Domain() string    { return r.namespace }
func (r Name) Namespace() string { return r.namespace }
func (r Name) Nick() string      { return r.nick }
func (r Name) Tag() string       { return r.tag }
func (r Name) Build() string     { return r.build }

// ParseName parses a model path string into a Name.
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
func ParseName(s string) Name {
	var r Name
	for kind, part := range NameParts(s) {
		switch kind {
		case Registry:
			r.domain = part
		case Namespace:
			r.namespace = part
		case Nick:
			r.nick = part
		case Tag:
			r.tag = part
		case Build:
			r.build = strings.ToUpper(part)
		case Invalid:
			return Name{}
		}
	}
	if !r.Valid() {
		return Name{}
	}
	return r
}

// Merge performs a partial merge of dst into src. Only the non-name parts
// are merged. The name part is always left untouched. Other parts are
// merged if and only if they are missing in dst.
//
// Use this for merging a fully qualified ref with a partial ref, such as
// when filling in a missing parts with defaults.
//
// The returned Name will only be valid if dst is valid.
func Merge(dst, src Name) Name {
	return Name{
		// name is left untouched
		nick: dst.nick,

		domain:    cmp.Or(dst.domain, src.domain),
		namespace: cmp.Or(dst.namespace, src.namespace),
		tag:       cmp.Or(dst.tag, src.tag),
		build:     cmp.Or(dst.build, src.build),
	}
}

// WithBuild returns a copy of r with the build set to the given string.
func (r Name) WithBuild(build string) Name {
	r.build = build
	return r
}

// Parts returns a sequence of the parts of a ref string from most specific
// to least specific.
//
// It normalizes the input string by removing "http://" and "https://" only.
// No other normalization is done.
//
// As a special case, question marks are ignored so they may be used as
// placeholders for missing parts in string literals.
func NameParts(s string) iter.Seq2[NamePart, string] {
	return func(yield func(NamePart, string) bool) {
		if strings.HasPrefix(s, "http://") {
			s = s[len("http://"):]
		}
		if strings.HasPrefix(s, "https://") {
			s = s[len("https://"):]
		}

		if len(s) > MaxNameLength || len(s) == 0 {
			return
		}

		yieldValid := func(kind NamePart, part string) bool {
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
					state, j = Nick, i
				default:
					yield(Invalid, "")
					return
				}
			case '/':
				switch state {
				case Nick, Tag, Build:
					if !yieldValid(Nick, s[i+1:j]) {
						return
					}
					state, j = Namespace, i
				case Namespace:
					if !yieldValid(Namespace, s[i+1:j]) {
						return
					}
					state, j = Registry, i
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
			yieldValid(Nick, s[:j])
		}
	}
}

// Valid returns true if the ref has a valid nick. To know if a ref is
// "complete", use Complete.
func (r Name) Valid() bool {
	// Parts ensures we only have valid parts, so no need to validate
	// them here, only check if we have a name or not.
	return r.nick != ""
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
