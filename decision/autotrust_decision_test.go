package decision

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func TestIndependentQwenChoiceCapacity(t *testing.T) {
	for _, encoding := range []string{"d3", "autotrust-decision"} {
		for _, count := range []int{1, 2, 255, 256, 257} {
			t.Run(fmt.Sprintf("%s/%d", encoding, count), func(t *testing.T) {
				var criteria []string
				for i := range count {
					criteria = append(criteria, fmt.Sprintf("%q:null", fmt.Sprintf("option-%03d", i)))
				}
				input := fmt.Sprintf(`{"model":"test","state":"x","questions":{"pick":{"type":"choice","instructions":"Pick one","criteria":{%s}}}}`, strings.Join(criteria, ","))
				var req Request
				if err := json.Unmarshal([]byte(input), &req); err != nil {
					t.Fatal(err)
				}
				compiled, err := Compile(req, encoding)
				minimum := 1
				if encoding == "autotrust-decision" {
					minimum = 2
				}
				if count < minimum || count > 255 {
					if err == nil {
						t.Fatalf("accepted %d options", count)
					}
					return
				}
				if err != nil {
					t.Fatalf("rejected %d options: %v", count, err)
				}
				if got := len(compiled.Request.Rows[0].Question.Options); got != count {
					t.Fatalf("compiled %d of %d options", got, count)
				}
			})
		}
	}
}

func TestIndependentQwenScoreLevels(t *testing.T) {
	for _, encoding := range []string{"d3", "autotrust-decision"} {
		for _, count := range []int{1, 2, 5, 6, 7, 10, 11} {
			t.Run(fmt.Sprintf("%s/%d", encoding, count), func(t *testing.T) {
				levels := make([]string, count)
				for i := range levels {
					levels[i] = fmt.Sprintf("%q", fmt.Sprintf("level-%d", i))
				}
				input := fmt.Sprintf(`{"model":"test","state":"x","questions":{"rating":{"type":"score","criteria":[%s]}}}`, strings.Join(levels, ","))
				var req Request
				if err := json.Unmarshal([]byte(input), &req); err != nil {
					t.Fatal(err)
				}
				_, err := Compile(req, encoding)
				valid := count >= 2 && count <= 10
				if encoding == "autotrust-decision" {
					valid = count == 6
				}
				if (err == nil) != valid {
					t.Fatalf("score levels %d: unexpected error %v", count, err)
				}
			})
		}
	}
}
