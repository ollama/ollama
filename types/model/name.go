package model

import (
	"cmp"
	"errors"
	"fmt"
	"hash/maphash"
	"io"
	"log/slog"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/ollama/ollama/types/structs"
)

// Errors
var (
	// ErrInvalidName, ErrIncompleteName, and ErrInvalidDigest are not
	// used by this package, but are exported so that other packages can
	// use them, instead of defining their own errors for them.
	ErrInvalidName    = errors.New("invalid model name")
	ErrIncompleteName = errors.New("incomplete model name")
	ErrInvalidDigest  = errors.New("invalid digest")
)

// Defaults
const (
	// MaskDefault is the default mask used by [Name.DisplayShortest].
	MaskDefault = "registry.ollama.ai/library/?:latest"

	// MaskNothing is a mask that masks nothing.
	MaskNothing = "?/?/?:?"

	// DefaultFill is the default fill used by [ParseName].
	FillDefault = "registry.ollama.ai/library/?:latest+Q4_0"

	// FillNothing is a fill that fills nothing.
	FillNothing = "?/?/?:?+?"
)

const MaxNamePartLen = 128

type PartKind int

// Levels of concreteness
const (
	// Each value aligns with its index in the Name.parts array.

	PartHost PartKind = iota
	PartNamespace
	PartModel
	PartTag
	PartBuild
	PartDigest

	// NumParts is the number of parts in a Name. In this list, it must
	// follow the final part.
	NumParts

	PartExtraneous = -1
)

var kindNames = map[PartKind]string{
	PartHost:      "Host",
	PartNamespace: "Namespace",
	PartModel:     "Name",
	PartTag:       "Tag",
	PartBuild:     "Build",
	PartDigest:    "Digest",
}

func (k PartKind) String() string {
	return cmp.Or(kindNames[k], "Unknown")
}

// Name is an opaque reference to a model. It holds the parts of a model
// with the case preserved, but is not directly comparable with other Names
// since model names can be represented with different casing depending on
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
// Name is complete, use [Name.IsComplete].
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
// The parts can be obtained in their original form by calling [Name.Parts].
//
// To check if a Name has at minimum a valid model part, use [Name.IsValid].
type Name struct {
	_     structs.Incomparable
	parts [NumParts]string // host, namespace, model, tag, build, digest

	// TODO(bmizerany): track offsets and hold s (raw string) here? We
	// could pack the offsets all into a single uint64 since the first
	// parts take less bits since their max offset is less than the max
	// offset of the next part. This would save a ton of bytes per Name
	// and mean zero allocations for String.
}

// ParseName parses s into a Name, and returns the result of filling it with
// defaults. The input string must be a valid string
// representation of a model name in the form:
//
//	[host/][namespace/]<model>[:tag][+build][@<digest-type>-<digest>]
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
//	"example.com/pdevine/thisisfine:7b+Q4_0@sha256-1234567890abcdef"
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
// # Fills
//
// For any valid s, the fill string is used to fill in missing parts of the
// Name. The fill string must be a valid Name with the exception that any part
// may be the string ("?"), which will not be considered for filling.
func ParseNameFill(s, fill string) Name {
	var r Name
	parts(s)(func(kind PartKind, part string) bool {
		if kind == PartDigest && !ParseDigest(part).IsValid() {
			r = Name{}
			return false
		}
		if kind == PartExtraneous || !IsValidNamePart(kind, part) {
			r = Name{}
			return false
		}
		r.parts[kind] = part
		return true
	})
	if r.IsValid() || r.IsResolved() {
		return fillName(r, fill)
	}
	return Name{}
}

// ParseName parses s into a Name, and returns the result of filling it
// with FillDefault. The input string must be a valid string representation
// of a model
func ParseName(s string) Name {
	return ParseNameFill(s, "")
}

func parseMask(s string) Name {
	var r Name
	parts(s)(func(kind PartKind, part string) bool {
		if part == "?" {
			// mask part; treat as empty but valid
			return true
		}
		if !IsValidNamePart(kind, part) {
			panic(fmt.Errorf("invalid mask part %s: %q", kind, part))
		}
		r.parts[kind] = part
		return true
	})
	return r
}

func MustParseName(s, fill string) Name {
	r := ParseNameFill(s, fill)
	if !r.IsValid() {
		panic("invalid Name: " + s)
	}
	return r
}

// fillName fills in the missing parts of dst with the parts of src.
//
// The returned Name will only be valid if dst is valid.
//
// It skipps fill parts that are "?".
func fillName(r Name, fill string) Name {
	fill = cmp.Or(fill, FillDefault)
	f := parseMask(fill)
	if fill != FillNothing && f.IsZero() {
		panic("invalid fill")
	}
	for i := range r.parts {
		if f.parts[i] == "?" {
			continue
		}
		r.parts[i] = cmp.Or(r.parts[i], f.parts[i])
	}
	return r
}

// WithBuild returns a copy of r with the build set to the given string.
func (r Name) WithBuild(build string) Name {
	r.parts[PartBuild] = build
	return r
}

func (r Name) WithDigest(digest Digest) Name {
	r.parts[PartDigest] = digest.String()
	return r
}

var mapHashSeed = maphash.MakeSeed()

// MapHash returns a case insensitive hash for use in maps and equality
// checks. For a convenient way to compare names, use [Name.EqualFold].
//
//nolint:errcheck
func (r Name) MapHash() uint64 {
	// correctly hash the parts with case insensitive comparison
	var h maphash.Hash
	h.SetSeed(mapHashSeed)
	for _, part := range r.parts {
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

func (r Name) slice(from, to PartKind) Name {
	var v Name
	copy(v.parts[from:to+1], r.parts[from:to+1])
	return v
}

// DisplayShortest returns the shortest possible, masked display string in form:
//
//	[host/][<namespace>/]<model>[:<tag>]
//
// # Masks
//
// The mask is a string that specifies which parts of the name to omit based
// on case-insensitive comparison. [Name.DisplayShortest] omits parts of the name
// that are the same as the mask, moving from left to right until the first
// unequal part is found. It then moves right to left until the first unequal
// part is found. The result is the shortest possible display string.
//
// Unlike a [Name] the mask can contain "?" characters which are treated as
// wildcards. A "?" will never match a part of the name, since a valid name
// can never contain a "?" character.
//
// For example: Given a Name ("registry.ollama.ai/library/mistral:latest") masked
// with ("registry.ollama.ai/library/?:latest") will produce the display string
// ("mistral").
//
// If mask is the empty string, then [MaskDefault] is used.
//
// DisplayShortest panics if the mask is not the empty string, MaskNothing, and
// invalid.
//
// # Builds
//
// For now, DisplayShortest does consider the build or return one in the
// result. We can lift this restriction when needed.
func (r Name) DisplayShortest(mask string) string {
	mask = cmp.Or(mask, MaskDefault)
	d := parseMask(mask)
	if mask != MaskNothing && r.IsZero() {
		panic("invalid Name")
	}
	for i := range PartTag {
		if !strings.EqualFold(r.parts[i], d.parts[i]) {
			break
		}
		r.parts[i] = ""
	}
	for i := PartTag; i >= 0; i-- {
		if !strings.EqualFold(r.parts[i], d.parts[i]) {
			break
		}
		r.parts[i] = ""
	}
	return r.slice(PartHost, PartTag).DisplayLong()
}

// DisplayLongest returns the result of r.DisplayShortest(MaskNothing).
func (r Name) DisplayLongest() string {
	return r.DisplayShortest(MaskNothing)
}

var seps = [...]string{
	PartHost:      "/",
	PartNamespace: "/",
	PartModel:     ":",
	PartTag:       "+",
	PartBuild:     "@",
	PartDigest:    "",
}

// WriteTo implements io.WriterTo. It writes the fullest possible display
// string in form:
//
//	<host>/<namespace>/<model>:<tag>+<build>@<digest-type>-<digest>
//
// Missing parts and their separators are not written.
//
// The full digest is always prefixed with "@". That is if [Name.IsValid]
// reports false and [Name.IsResolved] reports true, then the string is
// returned as "@<digest-type>-<digest>".
func (r Name) writeTo(w io.StringWriter) error {
	var partsWritten int
	for i := range r.parts {
		if r.parts[i] == "" {
			continue
		}
		if partsWritten > 0 || i == int(PartDigest) {
			if _, err := w.WriteString(seps[i-1]); err != nil {
				return err
			}
		}
		if _, err := w.WriteString(r.parts[i]); err != nil {
			return err
		}
		partsWritten++
	}
	return nil
}

var builderPool = sync.Pool{
	New: func() interface{} {
		return &strings.Builder{}
	},
}

// DisplayLong returns the fullest possible display string in form:
//
//	<host>/<namespace>/<model>:<tag>+<build>
//
// If any part is missing, it is omitted from the display string.
func (r Name) DisplayLong() string {
	b := builderPool.Get().(*strings.Builder)
	defer builderPool.Put(b)
	b.Reset()
	b.Grow(50) // arbitrarily long enough for most names
	_ = r.writeTo(b)
	return b.String()
}

// GoString implements fmt.GoStringer. It returns a string suitable for
// debugging and logging. It is similar to [Name.DisplayLong] but it always
// returns a string that includes all parts of the Name, with missing parts
// replaced with a ("?").
func (r Name) GoString() string {
	for i := range r.parts {
		r.parts[i] = cmp.Or(r.parts[i], "?")
	}
	return r.DisplayLong()
}

// LogValue implements slog.Valuer.
func (r Name) LogValue() slog.Value {
	return slog.StringValue(r.GoString())
}

// IsComplete reports whether the Name is fully qualified. That is it has a
// domain, namespace, name, tag, and build.
func (r Name) IsComplete() bool {
	return !slices.Contains(r.parts[:PartDigest], "")
}

// IsCompleteNoBuild is like [Name.IsComplete] but it does not require the
// build part to be present.
func (r Name) IsCompleteNoBuild() bool {
	return !slices.Contains(r.parts[:PartBuild], "")
}

// IsResolved reports true if the Name has a valid digest.
//
// It is possible to have a valid Name, or a complete Name that is not
// resolved.
func (r Name) IsResolved() bool {
	return r.Digest().IsValid()
}

// Digest returns the digest part of the Name, if any.
//
// If Digest returns a non-empty string, then [Name.IsResolved] will return
// true, and digest is considered valid.
func (r Name) Digest() Digest {
	// This was already validated by ParseName, so we can just return it.
	return Digest{r.parts[PartDigest]}
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
	return slices.CompareFunc(r.parts[:], o.parts[:], compareFold)
}

func compareFold(a, b string) int {
	return slices.CompareFunc([]rune(a), []rune(b), func(a, b rune) int {
		return cmp.Compare(downcase(a), downcase(b))
	})
}

func downcase(r rune) rune {
	if r >= 'A' && r <= 'Z' {
		return r - 'A' + 'a'
	}
	return r
}

func (r Name) Host() string      { return r.parts[PartHost] }
func (r Name) Namespace() string { return r.parts[PartNamespace] }
func (r Name) Model() string     { return r.parts[PartModel] }
func (r Name) Build() string     { return r.parts[PartBuild] }
func (r Name) Tag() string       { return r.parts[PartTag] }

// iter_Seq2 is a iter.Seq2 defined here to avoid the current build
// restrictions in the go1.22 iter package requiring the
// goexperiment.rangefunc tag to be set via the GOEXPERIMENT=rangefunc flag,
// which we are not yet ready to support.
//
// Once we are ready to support rangefunc, this can be removed and replaced
// with the iter.Seq2 type.
type iter_Seq2[A, B any] func(func(A, B) bool)

// Parts returns a sequence of the parts of a Name string from most specific
// to least specific.
//
// It normalizes the input string by removing "http://" and "https://" only.
// No other normalizations are performed.
func parts(s string) iter_Seq2[PartKind, string] {
	return func(yield func(PartKind, string) bool) {
		if strings.HasPrefix(s, "http://") {
			s = strings.TrimPrefix(s, "http://")
		} else {
			s = strings.TrimPrefix(s, "https://")
		}

		if len(s) > MaxNamePartLen || len(s) == 0 {
			return
		}

		numConsecutiveDots := 0
		partLen := 0
		state, j := PartDigest, len(s)
		for i := len(s) - 1; i >= 0; i-- {
			if partLen++; partLen > MaxNamePartLen {
				// catch a part that is too long early, so
				// we don't keep spinning on it, waiting for
				// an isInValidPart check which would scan
				// over it again.
				yield(state, s[i+1:j])
				return
			}

			switch s[i] {
			case '@':
				switch state {
				case PartDigest:
					if !yield(PartDigest, s[i+1:j]) {
						return
					}
					if i == 0 {
						// This is the form
						// "@<digest>" which is valid.
						//
						// We're done.
						return
					}
					state, j, partLen = PartBuild, i, 0
				default:
					yield(PartExtraneous, s[i+1:j])
					return
				}
			case '+':
				switch state {
				case PartBuild, PartDigest:
					if !yield(PartBuild, s[i+1:j]) {
						return
					}
					state, j, partLen = PartTag, i, 0
				default:
					yield(PartExtraneous, s[i+1:j])
					return
				}
			case ':':
				switch state {
				case PartTag, PartBuild, PartDigest:
					if !yield(PartTag, s[i+1:j]) {
						return
					}
					state, j, partLen = PartModel, i, 0
				case PartHost:
					// noop: support for host:port
				default:
					yield(PartExtraneous, s[i+1:j])
					return
				}
			case '/':
				switch state {
				case PartModel, PartTag, PartBuild, PartDigest:
					if !yield(PartModel, s[i+1:j]) {
						return
					}
					state, j = PartNamespace, i
				case PartNamespace:
					if !yield(PartNamespace, s[i+1:j]) {
						return
					}
					state, j, partLen = PartHost, i, 0
				default:
					yield(PartExtraneous, s[i+1:j])
					return
				}
			default:
				if s[i] == '.' {
					if numConsecutiveDots++; numConsecutiveDots > 1 {
						yield(state, "")
						return
					}
				} else {
					numConsecutiveDots = 0
				}
			}
		}

		if state <= PartNamespace {
			yield(state, s[:j])
		} else {
			yield(PartModel, s[:j])
		}
	}
}

func (r Name) IsZero() bool {
	return r.parts == [NumParts]string{}
}

// IsValid reports if a model has at minimum a valid model part.
func (r Name) IsValid() bool {
	// Parts ensures we only have valid parts, so no need to validate
	// them here, only check if we have a name or not.
	return r.parts[PartModel] != ""
}

// ParseNameFromURLPath parses forms of a URL path into a Name. Specifically,
// it trims any leading "/" and then calls [ParseName] with fill.
func ParseNameFromURLPath(s, fill string) Name {
	s = strings.TrimPrefix(s, "/")
	return ParseNameFill(s, fill)
}

func ParseNameFromURLPathFill(s, fill string) Name {
	return ParseNameFill(s, fill)
}

// URLPath returns a complete, canonicalized, relative URL path using the parts of a
// complete Name.
//
// The parts maintain their original case.
//
// Example:
//
//	ParseName("example.com/namespace/model:tag+build").URLPath() // returns "/example.com/namespace/model:tag"
func (r Name) DisplayURLPath() string {
	return r.DisplayShortest(MaskNothing)
}

// URLPath returns a complete, canonicalized, relative URL path using the parts of a
// complete Name in the form:
//
//	<host>/<namespace>/<model>/<tag>
//
// The parts are downcased.
func (r Name) URLPath() string {
	return strings.ToLower(path.Join(r.parts[:PartBuild]...))
}

// ParseNameFromFilepath parses a file path into a Name. The input string must be a
// valid file path representation of a model name in the form:
//
//	host/namespace/model/tag/build
//
// The zero valid is returned if s does not contain all path elements
// leading up to the model part, or if any path element is an invalid part
// for the its corresponding part kind.
//
// The fill string is used to fill in missing parts of any constructed Name.
// See [ParseName] for more information on the fill string.
func ParseNameFromFilepath(s, fill string) Name {
	var r Name
	for i := range PartBuild + 1 {
		part, rest, _ := strings.Cut(s, string(filepath.Separator))
		if !IsValidNamePart(i, part) {
			return Name{}
		}
		r.parts[i] = part
		s = rest
		if s == "" {
			break
		}
	}
	if s != "" {
		return Name{}
	}
	if !r.IsValid() {
		return Name{}
	}
	return fillName(r, fill)
}

// Filepath returns a complete, canonicalized, relative file path using the
// parts of a complete Name.
//
// Each parts is downcased, except for the build part which is upcased.
//
// Example:
//
//	ParseName("example.com/namespace/model:tag+build").Filepath() // returns "example.com/namespace/model/tag/BUILD"
func (r Name) Filepath() string {
	for i := range r.parts {
		if PartKind(i) == PartBuild {
			r.parts[i] = strings.ToUpper(r.parts[i])
		} else {
			r.parts[i] = strings.ToLower(r.parts[i])
		}
	}
	return filepath.Join(r.parts[:]...)
}

// FilepathNoBuild returns a complete, canonicalized, relative file path using
// the parts of a complete Name, but without the build part.
func (r Name) FilepathNoBuild() string {
	for i := range PartBuild {
		r.parts[i] = strings.ToLower(r.parts[i])
	}
	return filepath.Join(r.parts[:PartBuild]...)
}

// IsValidNamePart reports if s contains all valid characters for the given
// part kind and is under MaxNamePartLen bytes.
func IsValidNamePart(kind PartKind, s string) bool {
	if len(s) > MaxNamePartLen {
		return false
	}
	if s == "" {
		return false
	}
	var consecutiveDots int
	for _, c := range []byte(s) {
		if c == '.' {
			if consecutiveDots++; consecutiveDots >= 2 {
				return false
			}
		} else {
			consecutiveDots = 0
		}
		if !isValidByteFor(kind, c) {
			return false
		}
	}
	return true
}

func isValidByteFor(kind PartKind, c byte) bool {
	if kind == PartNamespace && c == '.' {
		return false
	}
	if kind == PartHost && c == ':' {
		return true
	}
	if c == '.' || c == '-' {
		return true
	}
	if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' {
		return true
	}
	return false
}
