package cmd

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"

	"github.com/ollama/ollama/api"
)

func TestExtractFilenames(t *testing.T) {
	// Unix style paths
	input := ` some preamble 
 ./relative\ path/one.png inbetween1 ./not a valid two.jpg inbetween2
/unescaped space /three.jpeg inbetween3 /valid\ path/dir/four.png "./quoted with spaces/five.svg`
	res := extractFileNames(input)
	assert.Len(t, res, 5)
	assert.Contains(t, res[0], "one.png")
	assert.Contains(t, res[1], "two.jpg")
	assert.Contains(t, res[2], "three.jpeg")
	assert.Contains(t, res[3], "four.png")
	assert.Contains(t, res[4], "five.svg")
	assert.NotContains(t, res[4], '"')
	assert.NotContains(t, res, "inbtween")

	// Windows style paths
	input = ` some preamble
 c:/users/jdoe/one.png inbetween1 c:/program files/someplace/two.jpg inbetween2 
 /absolute/nospace/three.jpeg inbetween3 /absolute/with space/four.png inbetween4
./relative\ path/five.svg inbetween5 "./relative with/spaces/six.png inbetween6
d:\path with\spaces\seven.svg inbetween7 c:\users\jdoe\eight.png inbetween8 
 d:\program files\someplace\nine.png inbetween9 "E:\program files\someplace\ten.svg some ending
`
	res = extractFileNames(input)
	assert.Len(t, res, 10)
	assert.NotContains(t, res, "inbtween")
	assert.Contains(t, res[0], "one.png")
	assert.Contains(t, res[0], "c:")
	assert.Contains(t, res[1], "two.jpg")
	assert.Contains(t, res[1], "c:")
	assert.Contains(t, res[2], "three.jpeg")
	assert.Contains(t, res[3], "four.png")
	assert.Contains(t, res[4], "five.svg")
	assert.Contains(t, res[5], "six.png")
	assert.Contains(t, res[6], "seven.svg")
	assert.Contains(t, res[6], "d:")
	assert.Contains(t, res[7], "eight.png")
	assert.Contains(t, res[7], "c:")
	assert.Contains(t, res[8], "nine.png")
	assert.Contains(t, res[8], "d:")
	assert.Contains(t, res[9], "ten.svg")
	assert.Contains(t, res[9], "E:")
}

func TestModelfileBuilder(t *testing.T) {
	opts := runOptions{
		Model:  "hork",
		System: "You are part horse and part shark, but all hork. Do horklike things",
		Messages: []api.Message{
			{Role: "user", Content: "Hey there hork!"},
			{Role: "assistant", Content: "Yes it is true, I am half horse, half shark."},
		},
		Options: map[string]any{
			"temperature":      0.9,
			"seed":             42,
			"penalize_newline": false,
			"stop":             []string{"hi", "there"},
		},
	}

	t.Run("model", func(t *testing.T) {
		expect := `FROM hork
SYSTEM You are part horse and part shark, but all hork. Do horklike things
PARAMETER penalize_newline false
PARAMETER seed 42
PARAMETER stop hi
PARAMETER stop there
PARAMETER temperature 0.9
MESSAGE user Hey there hork!
MESSAGE assistant Yes it is true, I am half horse, half shark.
`

		actual := buildModelfile(opts)
		if diff := cmp.Diff(expect, actual); diff != "" {
			t.Errorf("mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("parent model", func(t *testing.T) {
		opts.ParentModel = "horseshark"
		expect := `FROM horseshark
SYSTEM You are part horse and part shark, but all hork. Do horklike things
PARAMETER penalize_newline false
PARAMETER seed 42
PARAMETER stop hi
PARAMETER stop there
PARAMETER temperature 0.9
MESSAGE user Hey there hork!
MESSAGE assistant Yes it is true, I am half horse, half shark.
`
		actual := buildModelfile(opts)
		if diff := cmp.Diff(expect, actual); diff != "" {
			t.Errorf("mismatch (-want +got):\n%s", diff)
		}
	})
}
