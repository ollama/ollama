package blob

import (
	"cmp"
	"strings"
)

// Ref is an opaque reference to a blob.
//
// It is comparable and can be used as a map key.
//
// Users or Ref must check Valid before using it.
type Ref struct {
	name  string
	tag   string
	build string
}

// WithBuild returns a copy of r with the provided build. If the provided
// build is empty, it returns the short, unqualified copy of r.
func (r Ref) WithBuild(build string) Ref {
	if build == "" {
		return Ref{r.name, r.tag, ""}
	}
	if !isValidPart(build) {
		return Ref{}
	}
	return makeRef(r.name, r.tag, build)
}

// String returns the fully qualified ref string.
func (r Ref) String() string {
	var b strings.Builder
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

// Full returns the fully qualified ref string, or a string indicating the
// build is missing, or an empty string if the ref is invalid.
func (r Ref) Full() string {
	if !r.Valid() {
		return ""
	}
	return makeRef(r.name, r.tag, cmp.Or(r.build, "!(MISSING BUILD)")).String()
}

// Short returns the short ref string which does not include the build.
func (r Ref) Short() string {
	return r.WithBuild("").String()
}

func (r Ref) Valid() bool {
	return r.name != ""
}

func (r Ref) FullyQualified() bool {
	return r.name != "" && r.tag != "" && r.build != ""
}

func (r Ref) Name() string  { return r.name }
func (r Ref) Tag() string   { return r.tag }
func (r Ref) Build() string { return r.build }

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

	nameAndTag, build, expectBuild := strings.Cut(s, "+")
	name, tag, expectTag := strings.Cut(nameAndTag, ":")
	if !isValidPart(name) {
		return Ref{}
	}
	if expectTag && !isValidPart(tag) {
		return Ref{}
	}
	if expectBuild && !isValidPart(build) {
		return Ref{}
	}
	return makeRef(name, tag, build)
}

// makeRef makes a ref, skipping validation.
func makeRef(name, tag, build string) Ref {
	return Ref{name, cmp.Or(tag, "latest"), strings.ToUpper(build)}
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
