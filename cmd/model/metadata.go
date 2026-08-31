package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"math/big"
	"reflect"
	"slices"
	"strconv"
	"strings"
)

// decodeJSON preserves numbers and rejects duplicate keys: accepting the last
// occurrence could hide conflicting metadata in an otherwise plausible report.
func decodeJSON(data []byte) (any, error) {
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	v, err := jsonValue(d, 0)
	if err != nil {
		return nil, err
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, fmt.Errorf("trailing JSON data")
	}
	return v, nil
}

func jsonValue(d *json.Decoder, depth int) (any, error) {
	if depth > 100 {
		return nil, fmt.Errorf("JSON nesting exceeds 100 levels")
	}
	t, err := d.Token()
	if err != nil {
		return nil, err
	}
	switch t {
	case json.Delim('{'):
		m := make(map[string]any)
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return nil, err
			}
			k, ok := key.(string)
			if !ok {
				return nil, fmt.Errorf("invalid JSON object key")
			}
			if _, ok := m[k]; ok {
				return nil, fmt.Errorf("duplicate JSON key %q", k)
			}
			v, err := jsonValue(d, depth+1)
			if err != nil {
				return nil, err
			}
			m[k] = v
		}
		_, err := d.Token()
		return m, err
	case json.Delim('['):
		a := make([]any, 0)
		for d.More() {
			v, err := jsonValue(d, depth+1)
			if err != nil {
				return nil, err
			}
			a = append(a, v)
		}
		_, err := d.Token()
		return a, err
	default:
		return t, nil
	}
}

// MetadataChange uses JSON pointers. Presence bits distinguish null from absent.
type MetadataChange struct {
	Path         string
	LeftPresent  bool
	RightPresent bool
	Left         any
	Right        any
}

// proseMetadata keeps source line boundaries for useful human diffs while
// comparing prose independently of whitespace and line wrapping.
type proseMetadata struct {
	text  string
	lines []string
}

func normalizeProse(s string) proseMetadata {
	s = strings.ReplaceAll(strings.ReplaceAll(s, "\r\n", "\n"), "\r", "\n")
	lines := make([]string, 0, strings.Count(s, "\n")+1)
	for line := range strings.SplitSeq(s, "\n") {
		line = strings.Join(strings.Fields(line), " ")
		if line != "" {
			lines = append(lines, line)
		}
	}
	return proseMetadata{text: strings.Join(lines, " "), lines: lines}
}

func metadataDiff(path string, a, b any, ap, bp bool, out *[]MetadataChange) {
	if ap && bp {
		am, aok := a.(map[string]any)
		bm, bok := b.(map[string]any)
		if aok && bok {
			for _, k := range unionKeys(am, bm) {
				av, ax := am[k]
				bv, bx := bm[k]
				metadataDiff(path+"/"+pointerEscape(k), av, bv, ax, bx, out)
			}
			return
		}
		aa, aok := a.([]any)
		ba, bok := b.([]any)
		if aok && bok {
			for i := range max(len(aa), len(ba)) {
				var av, bv any
				if i < len(aa) {
					av = aa[i]
				}
				if i < len(ba) {
					bv = ba[i]
				}
				metadataDiff(path+"/"+strconv.Itoa(i), av, bv, i < len(aa), i < len(ba), out)
			}
			return
		}
		if metadataEqual(a, b) {
			return
		}
	}
	*out = append(*out, MetadataChange{path, ap, bp, a, b})
}

func metadataEqual(a, b any) bool {
	ap, aok := a.(proseMetadata)
	bp, bok := b.(proseMetadata)
	if aok && bok {
		return ap.text == bp.text
	}
	an, aok := a.(json.Number)
	bn, bok := b.(json.Number)
	if aok && bok {
		if an == bn {
			return true
		}
		// Ordinary config numbers compare exactly (including 1 vs 1.0), without
		// passing uint64 values through float64. Bound exponents before big.Rat
		// allocation; extreme representations remain exact lexical comparisons.
		ar, aok := decimalRat(string(an))
		br, bok := decimalRat(string(bn))
		return aok && bok && ar.Cmp(br) == 0
	}
	return reflect.DeepEqual(a, b)
}

func decimalRat(s string) (*big.Rat, bool) {
	if len(s) > 1024 {
		return nil, false
	}
	if i := strings.IndexAny(s, "eE"); i >= 0 {
		e, err := strconv.Atoi(s[i+1:])
		if err != nil || e < -1024 || e > 1024 {
			return nil, false
		}
	}
	return new(big.Rat).SetString(s)
}

func pointerEscape(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "~", "~0"), "/", "~1")
}

func unionKeys[V any](a, b map[string]V) []string {
	keys := slices.AppendSeq(slices.Collect(maps.Keys(a)), maps.Keys(b))
	slices.Sort(keys)
	return slices.Compact(keys)
}
