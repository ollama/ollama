package parser

import (
	"bytes"
	"strings"
	"testing"
)

func FuzzParseFile(f *testing.F) {
	f.Add("FROM foo\n")
	f.Add("FROM foo\nPARAMETER temperature 0.7\n")
	f.Add("FROM foo\nTEMPLATE \"\"\"{{ .Prompt }}\"\"\"\n")
	f.Add("FROM foo\nMESSAGE user hi\nMESSAGE assistant yo\n")
	f.Add("FROM foo\nSYSTEM \"you are helpful\"\n")
	f.Add("# comment\nFROM foo\nADAPTER ./x\n")
	f.Add("FROM foo\nLICENSE \"\"\"multi\nline\"\"\"\n")
	f.Add("")

	f.Fuzz(func(t *testing.T, s string) {
		if len(s) > 1<<16 {
			t.Skip()
		}
		mf, err := ParseFile(strings.NewReader(s))
		if err != nil {
			return
		}

		// Round-trip oracle: a successfully parsed Modelfile must serialize to
		// something that parses back to the same commands.
		out := mf.String()
		mf2, err := ParseFile(bytes.NewReader([]byte(out)))
		if err != nil {
			t.Fatalf("re-parse of String() failed: %v\n--- input ---\n%q\n--- String() ---\n%q", err, s, out)
		}
		if len(mf.Commands) != len(mf2.Commands) {
			t.Fatalf("round-trip changed command count: %d -> %d\n%q\n%q", len(mf.Commands), len(mf2.Commands), s, out)
		}
		for i := range mf.Commands {
			if mf.Commands[i] != mf2.Commands[i] {
				t.Fatalf("round-trip changed command %d: %+v -> %+v", i, mf.Commands[i], mf2.Commands[i])
			}
		}
	})
}
