package model

import (
	"bytes"
	"cmp"
	"errors"
	"hash/maphash"
	"io"
	"iter"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"unsafe"

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

type NamePart int

var kindNames = map[NamePart]string{
	Invalid:   "Invalid",
	Host:      "Host",
	Namespace: "Namespace",
	Model:     "Name",
	Tag:       "Tag",
	Build:     "Build",
}

func (k NamePart) String() string {
	return cmp.Or(kindNames[k], "Unknown")
}

// Levels of concreteness
const (
	Host NamePart = iota
	Namespace
	Model
	Tag
	Build

	NumParts = Build + 1

	Invalid = NamePart(-1)
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
	_     structs.Incomparable
	parts [NumParts]string
}

// ParseName parses s into a Name. The input string must be a valid string
// representation of a model name in the form:
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
//	"example.com/library/mistral:7b+x"
//	"example.com/eva/mistral:7b+Q4_0"
//	"mistral:7b+x"
//	"example.com/mike/mistral:latest+Q4_0"
//	"example.com/bruce/mistral:latest"
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
		if kind == Invalid {
			return Name{}
		}
		r.parts[kind] = part
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
	var r Name
	for i := range r.parts {
		r.parts[i] = cmp.Or(dst.parts[i], src.parts[i])
	}
	return r
}

// WithBuild returns a copy of r with the build set to the given string.
func (r Name) WithBuild(build string) Name {
	r.parts[Build] = build
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

func (r Name) slice(from, to NamePart) Name {
	var v Name
	copy(v.parts[from:to+1], r.parts[from:to+1])
	return v
}

// DisplayModel returns the a display string composed of the model only.
func (r Name) DisplayModel() string {
	return r.parts[Model]
}

// DisplayFullest returns the fullest possible display string in form:
//
//	<host>/<namespace>/<model>:<tag>
//
// If any part is missing, it is omitted from the display string.
//
// It does not include the build part. For the fullest possible display
// string with the build, use [Name.String].
func (r Name) DisplayFullest() string {
	return r.slice(Host, Tag).String()
}

// DisplayShort returns the fullest possible display string in form:
//
//	<model>:<tag>
//
// If any part is missing, it is omitted from the display string.
func (r Name) DisplayShort() string {
	return r.slice(Model, Tag).String()
}

// DisplayLong returns the fullest possible display string in form:
//
//	<namespace>/<model>:<tag>
//
// If any part is missing, it is omitted from the display string.
func (r Name) DisplayLong() string {
	return r.slice(Namespace, Tag).String()
}

var seps = [...]string{
	Host:      "/",
	Namespace: "/",
	Model:     ":",
	Tag:       "+",
	Build:     "",
}

func (r Name) WriteTo(w io.Writer) (n int64, err error) {
	for i := range r.parts {
		if r.parts[i] == "" {
			continue
		}
		if n > 0 {
			n1, err := io.WriteString(w, seps[i-1])
			n += int64(n1)
			if err != nil {
				return n, err
			}
		}
		n1, err := io.WriteString(w, r.parts[i])
		n += int64(n1)
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

var builderPool = sync.Pool{
	New: func() interface{} {
		return &strings.Builder{}
	},
}

// String returns the fullest possible display string in form:
//
//	<host>/<namespace>/<model>:<tag>+<build>
//
// If any part is missing, it is omitted from the display string.
//
// For the fullest possible display string without the build, use
// [Name.DisplayFullest].
func (r Name) String() string {
	b := builderPool.Get().(*strings.Builder)
	defer builderPool.Put(b)
	b.Reset()
	b.Grow(50) // arbitrarily long enough for most names
	_, _ = r.WriteTo(b)
	return b.String()
}

// GoString implements fmt.GoStringer. It returns a string suitable for
// debugging and logging. It is similar to [Name.String] but it always
// returns a string that includes all parts of the Name, with missing parts
// replaced with a ("?").
func (r Name) GoString() string {
	var v Name
	for i := range r.parts {
		v.parts[i] = cmp.Or(r.parts[i], "?")
	}
	return v.String()
}

// LogValue implements slog.Valuer.
func (r Name) LogValue() slog.Value {
	return slog.StringValue(r.GoString())
}

var bufPool = sync.Pool{
	New: func() interface{} {
		return new(bytes.Buffer)
	},
}

// MarshalText implements encoding.TextMarshaler.
func (r Name) MarshalText() ([]byte, error) {
	b := bufPool.Get().(*bytes.Buffer)
	b.Reset()
	b.Grow(50) // arbitrarily long enough for most names
	defer bufPool.Put(b)
	_, err := r.WriteTo(b)
	if err != nil {
		return nil, err
	}
	// TODO: We can remove this alloc if/when
	// https://github.com/golang/go/issues/62384 lands.
	return b.Bytes(), nil
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (r *Name) UnmarshalText(text []byte) error {
	// unsafeString is safe here because the contract of UnmarshalText
	// that text belongs to us for the duration of the call.
	*r = ParseName(unsafeString(text))
	return nil
}

func unsafeString(b []byte) string {
	return *(*string)(unsafe.Pointer(&b))
}

// Complete reports whether the Name is fully qualified. That is it has a
// domain, namespace, name, tag, and build.
func (r Name) Complete() bool {
	return !slices.Contains(r.parts[:], "")
}

// CompleteNoBuild is like [Name.Complete] but it does not require the
// build part to be present.
func (r Name) CompleteNoBuild() bool {
	return !slices.Contains(r.parts[:Build-1], "")
}

// EqualFold reports whether r and o are equivalent model names, ignoring
// case.
func (r Name) EqualFold(o Name) bool {
	return r.CompareFold(o) == 0
}

// CompareFold performs a case-insensitive cmp.Compare on r and o.
//
// This can be used with [slices.SortFunc].
//
// For simple equality checks, use [Name.EqualFold].
func (r Name) CompareFold(o Name) int {
	for i := range r.parts {
		if n := compareFold(r.parts[i], o.parts[i]); n != 0 {
			return n
		}
	}
	return 0
}

func compareFold(a, b string) int {
	// fast-path for unequal lengths
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

// TODO(bmizerany): driver.Value? (MarshalText etc should be enough)

// Parts returns the parts of the Name in order of concreteness.
//
// The length of the returned slice is always 5.
func (r Name) Parts() []string {
	return slices.Clone(r.parts[:])
}

// Parts returns a sequence of the parts of a Name string from most specific
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

		if len(s) > MaxNamePartLen || len(s) == 0 {
			return
		}

		yieldValid := func(kind NamePart, part string) bool {
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
				if !isValidByte(state, s[i]) {
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
	return r.parts[Model] != ""
}

// isValidPart returns true if given part is valid ascii [a-zA-Z0-9_\.-]
func isValidPart(kind NamePart, s string) bool {
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

func isValidByte(kind NamePart, c byte) bool {
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

func sumLens(a []string) (sum int) {
	for _, n := range a {
		sum += len(n)
	}
	return
}
