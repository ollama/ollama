// Copyright (c) Tailscale Inc & AUTHORS
// SPDX-License-Identifier: BSD-3-Clause

// Package stringsx provides additional string manipulation functions
// that aren't in the standard library's strings package or go4.org/mem.
package stringsx

import (
	"unicode"
	"unicode/utf8"
)

// CompareFold returns -1, 0, or 1 depending on whether a < b, a == b, or a > b,
// like cmp.Compare, but case insensitively.
func CompareFold(a, b string) int {
	// Track our position in both strings
	ia, ib := 0, 0
	for ia < len(a) && ib < len(b) {
		ra, wa := nextRuneLower(a[ia:])
		rb, wb := nextRuneLower(b[ib:])
		if ra < rb {
			return -1
		}
		if ra > rb {
			return 1
		}
		ia += wa
		ib += wb
		if wa == 0 || wb == 0 {
			break
		}
	}

	// If we've reached here, one or both strings are exhausted
	// The shorter string is "less than" if they match up to this point
	switch {
	case ia == len(a) && ib == len(b):
		return 0
	case ia == len(a):
		return -1
	default:
		return 1
	}
}

// nextRuneLower returns the next rune in the string, lowercased, along with its
// original (consumed) width in bytes. If the string is empty, it returns
// (utf8.RuneError, 0)
func nextRuneLower(s string) (r rune, width int) {
	r, width = utf8.DecodeRuneInString(s)
	return unicode.ToLower(r), width
}
