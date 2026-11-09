package d1

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestPyJSONIndent2(t *testing.T) {
	// Expected values are Python's json.dumps(json.loads(raw), ensure_ascii=False, indent=2).
	for _, tc := range []struct{ raw, want string }{
		{`{"b": 1, "a": [true, null, 1.5e-7], "b": "\u00e9\n<|im_end|>"}`, "{\n  \"b\": \"é\\n<|im_end|>\",\n  \"a\": [\n    true,\n    null,\n    1.5e-07\n  ]\n}"},
		{`[]`, "[]"},
		{`{}`, "{}"},
		{`{"n": 100.0, "big": 12345678901234567890, "neg": -0, "e": 1E+22, "f": 0.1}`, "{\n  \"n\": 100.0,\n  \"big\": 12345678901234567890,\n  \"neg\": 0,\n  \"e\": 1e+22,\n  \"f\": 0.1\n}"},
		{`[[1,[2]],{"k":{}}]`, "[\n  [\n    1,\n    [\n      2\n    ]\n  ],\n  {\n    \"k\": {}\n  }\n]"},
		{`"plain \u2028 text"`, "\"plain \u2028 text\""},
	} {
		got, err := pyJSONIndent2(json.RawMessage(tc.raw))
		if err != nil || got != tc.want {
			t.Errorf("%s:\ngot  %q (%v)\nwant %q", tc.raw, got, err, tc.want)
		}
	}
	deep := strings.Repeat("[", 9999) + strings.Repeat("]", 9999)
	for _, bad := range []string{``, `{`, `[1,]`, `{"a": 1} 2`, `NaN`, deep} {
		if _, err := pyJSONIndent2(json.RawMessage(bad)); err == nil {
			t.Errorf("%.40q: expected an error", bad)
		}
	}
}
