package model

import (
	"cmp"
	"errors"
	"hash/maphash"
	"iter"
	"log/slog"
	"slices"
	"strings"

	"github.com/ollama/ollama/x/types/structs"
)

// Errors
var (
	// ErrInvalidName is not used by this package, but is exported so that
	// other packages do not need to invent their own error type when they
	// need to return an error for an invalid name.
	ErrIncompleteName = errors.New("incomplete model name")
)

const MaxNamePartLen = 128

type NamePartKind int

var kindNames = map[NamePartKind]string{
	Invalid:   "Invalid",
	Host:      "Host",
	Namespace: "Namespace",
	Model:     "Name",
	Tag:       "Tag",
	Build:     "Build",
}

func (k NamePartKind) String() string {
	return cmp.Or(kindNames[k], "Unknown")
}

// Levels of concreteness
const (
	Invalid NamePartKind = iota
	Host
	Namespace
	Model
	Tag
	Build
)

// Name is an opaque reference to a model. It holds the parts of a model
// with the case preserved, but is not directly comparable with other Names
// since model names can be represented with different caseing depending on
// the use case. For instance, "Mistral" and "mistral" are the same model
// but each version may have come from different sources (e.g. copied from a
// Web page, or from a file path).
//
// Valid Names can ONLY be constructed by calling [ParseName].
//
// A Name is valid if and only if is have a valid Model part. The other parts
// are optional.
//
// A Name is considered "complete" if it has all parts present. To check if a
// Name is complete, use [Name.Complete].
//
// To compare two names in a case-insensitive manner, use [Name.EqualFold].
//
// The parts of a Name are:
//
//   - Host: the domain of the model (optional)
//   - Namespace: the namespace of the model (optional)
//   - Model: the name of the model (required)
//   - Tag: the tag of the model (optional)
//   - Build: the build of the model; usually the quantization or "file type" (optional)
//
// The parts can be obtained in their original form by calling [Name.Parts],
// [Name.Host], [Name.Namespace], [Name.Model], [Name.Tag], and [Name.Build].
//
// To check if a Name has at minimum a valid model part, use [Name.Valid].
//
// To check if a Name is fully qualified, use [Name.Complete]. A fully
// qualified name has all parts present.
//
// To update parts of a Name with defaults, use [Fill].
type Name struct {
	_ structs.Incomparable

	host      string
	namespace string
	model     string
	tag       string
	build     string
}

// ParseName parses s into a Name. The input string must be a valid form of
// a model name in the form:
//
//	<host>/<namespace>/<model>:<tag>+<build>
//
// The name part is required, all others are optional. If a part is missing,
// it is left empty in the returned Name. If a part is invalid, the zero Ref
// value is returned.
//
// The build part is normalized to uppercase.
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
//
// It returns the zero value if any part is invalid.
//
// As a rule of thumb, an valid name is one that can be round-tripped with
// the [Name.String] method. That means ("x+") is invalid because
// [Name.String] will not print a "+" if the build is empty.
func ParseName(s string) Name {
	var r Name
	for kind, part := range NameParts(s) {
		switch kind {
		case Host:
			r.host = part
		case Namespace:
			r.namespace = part
		case Model:
			r.model = part
		case Tag:
			r.tag = part
		case Build:
			r.build = part
		case Invalid:
			return Name{}
		}
	}
	if !r.Valid() {
		return Name{}
	}
	return r
}

// Fill fills in the missing parts of dst with the parts of src.
//
// The returned Name will only be valid if dst is valid.
func Fill(dst, src Name) Name {
	return Name{
		model:     cmp.Or(dst.model, src.model),
		host:      cmp.Or(dst.host, src.host),
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

var mapHashSeed = maphash.MakeSeed()

// MapHash returns a case insensitive hash for use in maps and equality
// checks. For a convienent way to compare names, use [Name.EqualFold].
func (r Name) MapHash() uint64 {
	// correctly hash the parts with case insensitive comparison
	var h maphash.Hash
	h.SetSeed(mapHashSeed)
	for _, part := range r.Parts() {
		// downcase the part for hashing
		for i := range part {
			c := part[i]
			if c >= 'A' && c <= 'Z' {
				c = c - 'A' + 'a'
			}
			h.WriteByte(c)
		}
	}
	return h.Sum64()
}

// DisplayModel returns the a display string of the model.
func (r Name) DisplayModel() string {
	return r.model
}

// DisplayComplete returns a complete display string of the Name. For any
// part that is missing, a ("?") is used. If a complete name is not required
// and only the longest possible name is needed, use [Name.String] or
// String or check if [Name.Complete] is true before calling.
//
// It does not include the build.
func (r Name) DisplayComplete() string {
	return (Name{
		host:      cmp.Or(r.host, "?"),
		namespace: cmp.Or(r.namespace, "?"),
		model:     cmp.Or(r.model, "?"),
		tag:       cmp.Or(r.tag, "?"),
	}).String()
}

// GoString implements fmt.GoStringer. It is like DisplayComplete but
// includes the build or a "?" if the build is missing.
func (r Name) GoString() string {
	return (Name{
		host:      cmp.Or(r.host, "?"),
		namespace: cmp.Or(r.namespace, "?"),
		model:     cmp.Or(r.model, "?"),
		tag:       cmp.Or(r.tag, "?"),
		build:     cmp.Or(r.build, "?"),
	}).String()
}

// LogValue implements slog.Valuer.
func (r Name) LogValue() slog.Value {
	return slog.StringValue(r.GoString())
}

// DisplayShort returns a short display string of the Name with only the
// model, tag, and build parts.
func (r Name) DisplayShort() string {
	return (Name{
		model: r.model,
		tag:   r.tag,
	}).String()
}

// DisplayLong returns a long display string of the Name including namespace,
// model, tag, and build parts.
func (r Name) DisplayLong() string {
	return (Name{
		namespace: r.namespace,
		model:     r.model,
		tag:       r.tag,
	}).String()
}

// String returns the fully qualified Name string.
func (r Name) String() string {
	var b strings.Builder
	if r.host != "" {
		b.WriteString(r.host)
		b.WriteString("/")
	}
	if r.namespace != "" {
		b.WriteString(r.namespace)
		b.WriteString("/")
	}
	b.WriteString(r.model)
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

// Complete reports whether the Name is fully qualified. That is it has a
// domain, namespace, name, tag, and build.
func (r Name) Complete() bool {
	return !slices.Contains(r.Parts(), "")
}

// EqualFold reports whether r and o are equivalent model names, ignoring
// case.
func (r Name) EqualFold(o Name) bool {
	return r.CompareFold(o) == 0
}

// CompareFold performs a case-insensitive comparison of two Names. It returns
// an integer comparing two Names lexicographically. The result will be 0 if
// r == o, -1 if r < o, and +1 if r > o.
//
// This can be used with [slice.SortFunc].
func (r Name) CompareFold(o Name) int {
	return cmp.Or(
		compareFold(r.host, o.host),
		compareFold(r.namespace, o.namespace),
		compareFold(r.model, o.model),
		compareFold(r.tag, o.tag),
		compareFold(r.build, o.build),
	)
}

func compareFold(a, b string) int {
	for i := 0; i < len(a) && i < len(b); i++ {
		ca, cb := downcase(a[i]), downcase(b[i])
		if n := cmp.Compare(ca, cb); n != 0 {
			return n
		}
	}
	return cmp.Compare(len(a), len(b))
}

func downcase(c byte) byte {
	if c >= 'A' && c <= 'Z' {
		return c + 'a' - 'A'
	}
	return c
}

// TODO(bmizerany): Compare
// TODO(bmizerany): MarshalText/UnmarshalText
// TODO(bmizerany): LogValue
// TODO(bmizerany): driver.Value? (MarshalText etc should be enough)

// Parts returns the parts of the Name in order of concreteness.
//
// The length of the returned slice is always 5.
func (r Name) Parts() []string {
	return []string{
		r.host,
		r.namespace,
		r.model,
		r.tag,
		r.build,
	}
}

// Parts returns a sequence of the parts of a Name string from most specific
// to least specific.
//
// It normalizes the input string by removing "http://" and "https://" only.
// No other normalization is done.
//
// As a special case, question marks are ignored so they may be used as
// placeholders for missing parts in string literals.
func NameParts(s string) iter.Seq2[NamePartKind, string] {
	return func(yield func(NamePartKind, string) bool) {
		if strings.HasPrefix(s, "http://") {
			s = s[len("http://"):]
		}
		if strings.HasPrefix(s, "https://") {
			s = s[len("https://"):]
		}

		if len(s) > MaxNamePartLen || len(s) == 0 {
			return
		}

		yieldValid := func(kind NamePartKind, part string) bool {
			if !isValidPart(kind, part) {
				yield(Invalid, "")
				return false
			}
			return yield(kind, part)
		}

		partLen := 0
		state, j := Build, len(s)
		for i := len(s) - 1; i >= 0; i-- {
			if partLen++; partLen > MaxNamePartLen {
				yield(Invalid, "")
				return
			}
			switch s[i] {
			case '+':
				switch state {
				case Build:
					if !yieldValid(Build, s[i+1:j]) {
						return
					}
					state, j, partLen = Tag, i, 0
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
					state, j, partLen = Model, i, 0
				default:
					yield(Invalid, "")
					return
				}
			case '/':
				switch state {
				case Model, Tag, Build:
					if !yieldValid(Model, s[i+1:j]) {
						return
					}
					state, j = Namespace, i
				case Namespace:
					if !yieldValid(Namespace, s[i+1:j]) {
						return
					}
					state, j, partLen = Host, i, 0
				default:
					yield(Invalid, "")
					return
				}
			default:
				if !isValidPart(state, s[i:i+1]) {
					yield(Invalid, "")
					return
				}
			}
		}

		if state <= Namespace {
			yieldValid(state, s[:j])
		} else {
			yieldValid(Model, s[:j])
		}
	}
}

// Valid returns true if the Name has a valid nick. To know if a Name is
// "complete", use Complete.
func (r Name) Valid() bool {
	// Parts ensures we only have valid parts, so no need to validate
	// them here, only check if we have a name or not.
	return r.model != ""
}

// isValidPart returns true if given part is valid ascii [a-zA-Z0-9_\.-]
func isValidPart(kind NamePartKind, s string) bool {
	if s == "" {
		return false
	}
	for _, c := range []byte(s) {
		if !isValidByte(kind, c) {
			return false
		}
	}
	return true
}

func isValidByte(kind NamePartKind, c byte) bool {
	if kind == Namespace && c == '.' {
		return false
	}
	if c == '.' || c == '-' {
		return true
	}
	if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' {
		return true
	}
	return false
}
