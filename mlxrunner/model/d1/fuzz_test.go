package d1

import (
	"encoding/json"
	"strings"
	"testing"
)

func FuzzParseQuestion(f *testing.F) {
	for _, seed := range []string{
		`{"type":"noul","instructions":"Refund?","criteria":{"true":"asks"}}`,
		`{"type":"choice","instructions":"Team?","criteria":{"a":"x","b":null}}`,
		`{"type":"score","instructions":"Urgent?","criteria":["low","high"]}`,
		`{"type":"noul","instructions":{"field":"payee"}}`,
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, raw string) {
		q, err := parseQuestion("q", json.RawMessage(raw))
		if err != nil {
			return
		}
		if strings.TrimSpace(q.said) == "" {
			t.Fatalf("accepted blank instructions: %s", raw)
		}
		switch q.typ {
		case "noul":
		case "choice":
			if len(q.options) < 2 || len(q.options) > maxOptions {
				t.Fatalf("accepted %d options: %s", len(q.options), raw)
			}
		case "score":
			if len(q.levels) < 2 || len(q.levels) > 10 {
				t.Fatalf("accepted %d levels: %s", len(q.levels), raw)
			}
		default:
			t.Fatalf("accepted type %q: %s", q.typ, raw)
		}
	})
}
