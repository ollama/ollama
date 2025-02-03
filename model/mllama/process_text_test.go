package mllama

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"github.com/ollama/ollama/model"
)

func TestProcessText(t *testing.T) {
	ours, err := model.New(filepath.Join("testdata", "model.bin"))
	if errors.Is(err, os.ErrNotExist) {
		t.Skip("no model.bin")
	} else if err != nil {
		t.Fatal(err)
	}

	t.Run("decode", func(t *testing.T) {
		f, err := os.Open(filepath.Join("testdata", "theirs.json"))
		if errors.Is(err, os.ErrNotExist) {
			t.Skip("no theirs.json")
		} else if err != nil {
			t.Fatal(err)
		}
		defer f.Close()

		var theirs [][]byte
		if err := json.NewDecoder(f).Decode(&theirs); err != nil {
			t.Fatal(err)
		}

		for id := range theirs {
			ids := []int32{int32(id)}
			s, err := ours.(model.TextProcessor).Decode(ids)
			if err != nil {
				t.Fatal(err)
			}

			if diff := cmp.Diff(string(theirs[id]), s); diff != "" {
				t.Errorf("%d no match (-theirs +ours):\n%s", id, diff)
			}
		}
	})

	t.Run("encode", func(t *testing.T) {
		f, err := os.Open(filepath.Join("..", "testdata", "inputs.json"))
		if errors.Is(err, os.ErrNotExist) {
			t.Skip("no inputs.json")
		} else if err != nil {
			t.Fatal(err)
		}
		defer f.Close()

		var inputs []struct {
			Values []byte  `json:"base64"`
			IDs    []int32 `json:"ids"`
		}

		if err := json.NewDecoder(f).Decode(&inputs); err != nil {
			t.Fatal(err)
		}

		for i, input := range inputs {
			if i == 45 {
				t.Skip("skip 45")
			}

			t.Run(strconv.Itoa(i), func(t *testing.T) {
				ids, err := ours.(model.TextProcessor).Encode(string(input.Values))
				if err != nil {
					t.Fatal(err)
				}

				if diff := cmp.Diff(input.IDs, ids, cmpopts.EquateEmpty()); diff != "" {
					t.Errorf("%s: no match (-theirs +ours):\n%s", input.Values, diff)
				}
			})
		}
	})
}
