package model

import (
	"bytes"
	"cmp"
	"database/sql"
	"database/sql/driver"
	"errors"
	"fmt"
	"hash/maphash"
	"io"
	"iter"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"unicode"

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

// Levels of concreteness
const (
	PartHost NamePart = iota
	PartNamespace
	PartModel
	PartTag
	PartBuild
	PartDigest

	// Invalid is a special part that is used to indicate that a part is
	// invalid. It is not a valid part of a Name.
	//
	// It should be kept as the last part in the list.
	PartInvalid

	NumParts = PartInvalid
)

var kindNames = map[NamePart]string{
	PartInvalid:   "Invalid",
	PartHost:      "Host",
	PartNamespace: "Namespace",
	PartModel:     "Name",
	PartTag:       "Tag",
	PartBuild:     "Build",
	PartDigest:    "Digest",
}

func (k NamePart) String() string {
	return cmp.Or(kindNames[k], "Unknown")
}

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
// The parts can be obtained in their original form by calling [Name.Parts].
//
// To check if a Name has at minimum a valid model part, use [Name.Valid].
//
// To make a Name by filling in missing parts from another Name, use [Fill].
type Name struct {
	_     structs.Incomparable
	parts [NumParts]string

	// TODO(bmizerany): track offsets and hold s (raw string) here? We
	// could pack the offests all into a single uint64 since the first
	// parts take less bits since their max offset is less than the max
	// offset of the next part. This would save a ton of bytes per Name
	// and mean zero allocations for String.
}

// ParseName parses s into a Name. The input string must be a valid string
// representation of a model name in the form:
//
//	<host>/<namespace>/<model>:<tag>+<build>@<digest-type>-<digest>
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
//	"example.com/mistral:7b+Q4_0@sha256-1234567890abcdef"
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
	for kind, part := range Parts(s) {
		if kind == PartInvalid {
			return Name{}
		}
		r.parts[kind] = part
	}
	if r.Valid() || r.Resolved() {
		return r
	}
	return Name{}
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
	r.parts[PartBuild] = build
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
	return r.parts[PartModel]
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
	return r.slice(PartHost, PartTag).String()
}

// DisplayShort returns the fullest possible display string in form:
//
//	<model>:<tag>
//
// If any part is missing, it is omitted from the display string.
func (r Name) DisplayShort() string {
	return r.slice(PartModel, PartTag).String()
}

// DisplayLong returns the fullest possible display string in form:
//
//	<namespace>/<model>:<tag>
//
// If any part is missing, it is omitted from the display string.
func (r Name) DisplayLong() string {
	return r.slice(PartNamespace, PartTag).String()
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
// Missing parts and their seperators are not written.
//
// The full digest is always prefixed with "@". That is if [Name.Valid]
// reports false and [Name.Resolved] reports true, then the string is
// returned as "@<digest-type>-<digest>".
func (r Name) WriteTo(w io.Writer) (n int64, err error) {
	for i := range r.parts {
		if r.parts[i] == "" {
			continue
		}
		if n > 0 || NamePart(i) == PartDigest {
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

// MarshalText implements [encoding.TextMarshaler].
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

// UnmarshalText implements [encoding.TextUnmarshaler].
//
// It is an error to call UnmarshalText on a valid Name.
func (r *Name) UnmarshalText(text []byte) error {
	if r.Valid() {
		// The invariant of UnmarshalText is that it should only be
		// called on an invalid/zero Name. If we allow UnmarshalText
		// on a valid Name, then the Name will be mutated, breaking
		// the immutability of the Name.
		return errors.New("model.Name: illegal UnmarshalText on valid Name")
	}

	// The contract of UnmarshalText  is that we copy to keep the text.
	*r = ParseName(string(text))
	return nil
}

var (
	_ driver.Valuer = Name{}
	_ sql.Scanner   = (*Name)(nil)
)

// Scan implements [database/sql.Scanner].
func (r *Name) Scan(src any) error {
	if r.Valid() {
		// The invariant of Scan is that it should only be called on an
		// invalid/zero Name. If we allow Scan on a valid Name, then the
		// Name will be mutated, breaking the immutability of the Name.
		return errors.New("model.Name: illegal Scan on valid Name")

	}
	switch v := src.(type) {
	case string:
		*r = ParseName(v)
		return nil
	case []byte:
		*r = ParseName(string(v))
		return nil
	}
	return errors.New("model.Name: invalid Scan source")
}

// Value implements [database/sql/driver.Valuer].
func (r Name) Value() (driver.Value, error) {
	return r.String(), nil
}

// Complete reports whether the Name is fully qualified. That is it has a
// domain, namespace, name, tag, and build.
func (r Name) Complete() bool {
	return !slices.Contains(r.parts[:PartDigest], "")
}

// CompleteNoBuild is like [Name.Complete] but it does not require the
// build part to be present.
func (r Name) CompleteNoBuild() bool {
	return !slices.Contains(r.parts[:PartBuild], "")
}

// Resolved reports true if the Name has a valid digest.
//
// It is possible to have a valid Name, or a complete Name that is not
// resolved.
func (r Name) Resolved() bool {
	return r.parts[PartDigest] != ""
}

// Digest returns the digest part of the Name, if any.
//
// If Digest returns a non-empty string, then [Name.Resolved] will return
// true, and digest is considered valid.
func (r Name) Digest() string {
	return r.parts[PartDigest]
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
func Parts(s string) iter.Seq2[NamePart, string] {
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
				yield(PartInvalid, "")
				return false
			}
			return yield(kind, part)
		}

		partLen := 0
		state, j := PartDigest, len(s)
		for i := len(s) - 1; i >= 0; i-- {
			if partLen++; partLen > MaxNamePartLen {
				yield(PartInvalid, "")
				return
			}

			switch s[i] {
			case '@':
				switch state {
				case PartDigest:
					part := s[i+1:]
					if isValidDigest(part) {
						if !yield(PartDigest, part) {
							return
						}
						if i == 0 {
							// The name is in
							// the form of
							// "@digest". This
							// is valid ans so
							// we want to skip
							// the final
							// validation for
							// any other state.
							return
						}
					} else {
						yield(PartInvalid, "")
						return
					}
					state, j, partLen = PartBuild, i, 0
				default:
					yield(PartInvalid, "")
					return
				}
			case '+':
				switch state {
				case PartBuild, PartDigest:
					if !yieldValid(PartBuild, s[i+1:j]) {
						return
					}
					state, j, partLen = PartTag, i, 0
				default:
					yield(PartInvalid, "")
					return
				}
			case ':':
				switch state {
				case PartTag, PartBuild, PartDigest:
					if !yieldValid(PartTag, s[i+1:j]) {
						return
					}
					state, j, partLen = PartModel, i, 0
				default:
					yield(PartInvalid, "")
					return
				}
			case '/':
				switch state {
				case PartModel, PartTag, PartBuild, PartDigest:
					if !yieldValid(PartModel, s[i+1:j]) {
						return
					}
					state, j = PartNamespace, i
				case PartNamespace:
					if !yieldValid(PartNamespace, s[i+1:j]) {
						return
					}
					state, j, partLen = PartHost, i, 0
				default:
					yield(PartInvalid, "")
					return
				}
			default:
				if !isValidByte(state, s[i]) {
					yield(PartInvalid, "")
					return
				}
			}
		}

		if state <= PartNamespace {
			yieldValid(state, s[:j])
		} else {
			yieldValid(PartModel, s[:j])
		}
	}
}

// Valid returns true if the Name hPartas a valid nick. To know if a Name is
// "complete", use [Name.Complete].
func (r Name) Valid() bool {
	// Parts ensures we only have valid parts, so no need to validate
	// them here, only check if we have a name or not.
	return r.parts[PartModel] != ""
}

// isValidPart returns Parttrue if given part is valid ascii [a-zA-Z0-9_\.-]
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
	if kind == PartNamespace && c == '.' {
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

// isValidDigest returns true if the given string in the form of
// "<digest-type>-<digest>", and <digest-type> is in the form of [a-z0-9]+
// and <digest> is a valid hex string.
//
// It does not check if the digest is a valid hash for the given digest
// type, or restrict the digest type to a known set of types. This is left
// up to ueers of this package.
func isValidDigest(s string) bool {
	typ, digest, ok := strings.Cut(s, "-")
	res := ok && isValidDigestType(typ) && isValidHex(digest)
	fmt.Printf("DEBUG: %q: typ: %s, digest: %s, ok: %v res: %v\n", s, typ, digest, ok, res)
	return res
}

func isValidDigestType(s string) bool {
	if len(s) == 0 {
		return false
	}
	for _, r := range s {
		if !unicode.IsLower(r) && !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

func isValidHex(s string) bool {
	if len(s) == 0 {
		return false
	}
	for i := range s {
		c := s[i]
		if c < '0' || c > '9' && c < 'a' || c > 'f' {
			return false
		}
	}
	return true
}
