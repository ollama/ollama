package convert

// sanitizeNonFiniteJSON rewrites non-standard JSON numeric tokens that some
// HF configs emit (Infinity, -Infinity, NaN) into standard JSON numbers.
//
// This is intentionally conservative:
// - only runs outside quoted strings
// - only rewrites full tokens
//
// We map these values to 0 because encoding/json rejects non-finite values,
// and these fields are typically model-side metadata not consumed by the
// converter.
func sanitizeNonFiniteJSON(in []byte) []byte {
	if len(in) == 0 {
		return in
	}

	out := make([]byte, 0, len(in))
	inString := false
	escape := false

	for i := 0; i < len(in); {
		c := in[i]

		if inString {
			out = append(out, c)
			if escape {
				escape = false
			} else if c == '\\' {
				escape = true
			} else if c == '"' {
				inString = false
			}
			i++
			continue
		}

		if c == '"' {
			inString = true
			out = append(out, c)
			i++
			continue
		}

		if hasToken(in, i, "-Infinity") {
			out = append(out, '0')
			i += len("-Infinity")
			continue
		}

		if hasToken(in, i, "Infinity") {
			out = append(out, '0')
			i += len("Infinity")
			continue
		}

		if hasToken(in, i, "NaN") {
			out = append(out, '0')
			i += len("NaN")
			continue
		}

		out = append(out, c)
		i++
	}

	return out
}

func hasToken(in []byte, at int, tok string) bool {
	end := at + len(tok)
	if at < 0 || end > len(in) {
		return false
	}
	if string(in[at:end]) != tok {
		return false
	}
	if at > 0 && !isJSONValuePrefixBoundary(in[at-1]) {
		return false
	}
	if end < len(in) && !isJSONValueSuffixBoundary(in[end]) {
		return false
	}
	return true
}

func isJSONWhitespace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

func isJSONValuePrefixBoundary(b byte) bool {
	return isJSONWhitespace(b) || b == ':' || b == ',' || b == '['
}

func isJSONValueSuffixBoundary(b byte) bool {
	return isJSONWhitespace(b) || b == ',' || b == ']' || b == '}'
}
