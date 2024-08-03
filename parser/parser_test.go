package parser

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"strings"
	"testing"
	"unicode/utf16"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/unicode"
)

func TestParseFileFile(t *testing.T) {
	input := `
FROM model1
ADAPTER adapter1
LICENSE MIT
PARAMETER param1 value1
PARAMETER param2 value2
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
`

	reader := strings.NewReader(input)

	modelfile, err := ParseFile(reader)
	require.NoError(t, err)

	expectedCommands := []Command{
		{Name: "model", Args: "model1"},
		{Name: "adapter", Args: "adapter1"},
		{Name: "license", Args: "MIT"},
		{Name: "param1", Args: "value1"},
		{Name: "param2", Args: "value2"},
		{Name: "template", Args: "{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n{{ .Response }}<|eot_id|>"},
	}

	assert.Equal(t, expectedCommands, modelfile.Commands)
}

func TestParseFileTrimSpace(t *testing.T) {
	input := `
FROM "     model 1"
ADAPTER      adapter3
LICENSE "MIT       "
PARAMETER param1        value1
PARAMETER param2    value2
TEMPLATE """   {{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>   """
`

	reader := strings.NewReader(input)

	modelfile, err := ParseFile(reader)
	require.NoError(t, err)

	expectedCommands := []Command{
		{Name: "model", Args: "     model 1"},
		{Name: "adapter", Args: "adapter3"},
		{Name: "license", Args: "MIT       "},
		{Name: "param1", Args: "value1"},
		{Name: "param2", Args: "value2"},
		{Name: "template", Args: "   {{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n{{ .Response }}<|eot_id|>   "},
	}

	assert.Equal(t, expectedCommands, modelfile.Commands)
}

func TestParseFileFrom(t *testing.T) {
	cases := []struct {
		input    string
		expected []Command
		err      error
	}{
		{
			"FROM \"FOO  BAR  \"",
			[]Command{{Name: "model", Args: "FOO  BAR  "}},
			nil,
		},
		{
			"FROM \"FOO BAR\"\nPARAMETER param1 value1",
			[]Command{{Name: "model", Args: "FOO BAR"}, {Name: "param1", Args: "value1"}},
			nil,
		},
		{
			"FROM     FOOO BAR    ",
			[]Command{{Name: "model", Args: "FOOO BAR"}},
			nil,
		},
		{
			"FROM /what/is/the path ",
			[]Command{{Name: "model", Args: "/what/is/the path"}},
			nil,
		},
		{
			"FROM foo",
			[]Command{{Name: "model", Args: "foo"}},
			nil,
		},
		{
			"FROM /path/to/model",
			[]Command{{Name: "model", Args: "/path/to/model"}},
			nil,
		},
		{
			"FROM /path/to/model/fp16.bin",
			[]Command{{Name: "model", Args: "/path/to/model/fp16.bin"}},
			nil,
		},
		{
			"FROM llama3:latest",
			[]Command{{Name: "model", Args: "llama3:latest"}},
			nil,
		},
		{
			"FROM llama3:7b-instruct-q4_K_M",
			[]Command{{Name: "model", Args: "llama3:7b-instruct-q4_K_M"}},
			nil,
		},
		{
			"", nil, errMissingFrom,
		},
		{
			"PARAMETER param1 value1",
			nil,
			errMissingFrom,
		},
		{
			"PARAMETER param1 value1\nFROM foo",
			[]Command{{Name: "param1", Args: "value1"}, {Name: "model", Args: "foo"}},
			nil,
		},
		{
			"PARAMETER what the \nFROM lemons make lemonade ",
			[]Command{{Name: "what", Args: "the"}, {Name: "model", Args: "lemons make lemonade"}},
			nil,
		},
	}

	for _, c := range cases {
		t.Run("", func(t *testing.T) {
			modelfile, err := ParseFile(strings.NewReader(c.input))
			require.ErrorIs(t, err, c.err)
			if modelfile != nil {
				assert.Equal(t, c.expected, modelfile.Commands)
			}
		})
	}
}

func TestParseFileParametersMissingValue(t *testing.T) {
	input := `
FROM foo
PARAMETER param1
`

	reader := strings.NewReader(input)

	_, err := ParseFile(reader)
	require.ErrorIs(t, err, io.ErrUnexpectedEOF)
}

func TestParseFileBadCommand(t *testing.T) {
	input := `
FROM foo
BADCOMMAND param1 value1
`
	_, err := ParseFile(strings.NewReader(input))
	require.ErrorIs(t, err, errInvalidCommand)
}

func TestParseFileMessages(t *testing.T) {
	cases := []struct {
		input    string
		expected []Command
		err      error
	}{
		{
			`
FROM foo
MESSAGE system You are a file parser. Always parse things.
`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "message", Args: "system: You are a file parser. Always parse things."},
			},
			nil,
		},
		{
			`
FROM foo
MESSAGE system You are a file parser. Always parse things.`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "message", Args: "system: You are a file parser. Always parse things."},
			},
			nil,
		},
		{
			`
FROM foo
MESSAGE system You are a file parser. Always parse things.
MESSAGE user Hey there!
MESSAGE assistant Hello, I want to parse all the things!
`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "message", Args: "system: You are a file parser. Always parse things."},
				{Name: "message", Args: "user: Hey there!"},
				{Name: "message", Args: "assistant: Hello, I want to parse all the things!"},
			},
			nil,
		},
		{
			`
FROM foo
MESSAGE system """
You are a multiline file parser. Always parse things.
"""
			`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "message", Args: "system: \nYou are a multiline file parser. Always parse things.\n"},
			},
			nil,
		},
		{
			`
FROM foo
MESSAGE badguy I'm a bad guy!
`,
			nil,
			errInvalidMessageRole,
		},
		{
			`
FROM foo
MESSAGE system
`,
			nil,
			io.ErrUnexpectedEOF,
		},
		{
			`
FROM foo
MESSAGE system`,
			nil,
			io.ErrUnexpectedEOF,
		},
	}

	for _, c := range cases {
		t.Run("", func(t *testing.T) {
			modelfile, err := ParseFile(strings.NewReader(c.input))
			require.ErrorIs(t, err, c.err)
			if modelfile != nil {
				assert.Equal(t, c.expected, modelfile.Commands)
			}
		})
	}
}

func TestParseFileQuoted(t *testing.T) {
	cases := []struct {
		multiline string
		expected  []Command
		err       error
	}{
		{
			`
FROM foo
SYSTEM """
This is a
multiline system.
"""
			`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "system", Args: "\nThis is a\nmultiline system.\n"},
			},
			nil,
		},
		{
			`
FROM foo
SYSTEM """
This is a
multiline system."""
			`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "system", Args: "\nThis is a\nmultiline system."},
			},
			nil,
		},
		{
			`
FROM foo
SYSTEM """This is a
multiline system."""
			`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "system", Args: "This is a\nmultiline system."},
			},
			nil,
		},
		{
			`
FROM foo
SYSTEM """This is a multiline system."""
			`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "system", Args: "This is a multiline system."},
			},
			nil,
		},
		{
			`
FROM foo
SYSTEM """This is a multiline system.""
			`,
			nil,
			io.ErrUnexpectedEOF,
		},
		{
			`
FROM foo
SYSTEM "
			`,
			nil,
			io.ErrUnexpectedEOF,
		},
		{
			`
FROM foo
SYSTEM """
This is a multiline system with "quotes".
"""
`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "system", Args: "\nThis is a multiline system with \"quotes\".\n"},
			},
			nil,
		},
		{
			`
FROM foo
SYSTEM """"""
`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "system", Args: ""},
			},
			nil,
		},
		{
			`
FROM foo
SYSTEM ""
`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "system", Args: ""},
			},
			nil,
		},
		{
			`
FROM foo
SYSTEM "'"
`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "system", Args: "'"},
			},
			nil,
		},
		{
			`
FROM foo
SYSTEM """''"'""'""'"'''''""'""'"""
`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "system", Args: `''"'""'""'"'''''""'""'`},
			},
			nil,
		},
		{
			`
FROM foo
TEMPLATE """
{{ .Prompt }}
"""`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "template", Args: "\n{{ .Prompt }}\n"},
			},
			nil,
		},
	}

	for _, c := range cases {
		t.Run("", func(t *testing.T) {
			modelfile, err := ParseFile(strings.NewReader(c.multiline))
			require.ErrorIs(t, err, c.err)
			if modelfile != nil {
				assert.Equal(t, c.expected, modelfile.Commands)
			}
		})
	}
}

func TestParseFileParameters(t *testing.T) {
	cases := map[string]struct {
		name, value string
	}{
		"numa true":                    {"numa", "true"},
		"num_ctx 1":                    {"num_ctx", "1"},
		"num_batch 1":                  {"num_batch", "1"},
		"num_gqa 1":                    {"num_gqa", "1"},
		"num_gpu 1":                    {"num_gpu", "1"},
		"main_gpu 1":                   {"main_gpu", "1"},
		"low_vram true":                {"low_vram", "true"},
		"cache_type_k f16":             {"cache_type_k", "f16"},
		"cache_type_v f16":             {"cache_type_v", "f16"},
		"logits_all true":              {"logits_all", "true"},
		"vocab_only true":              {"vocab_only", "true"},
		"use_mmap true":                {"use_mmap", "true"},
		"use_mlock true":               {"use_mlock", "true"},
		"num_thread 1":                 {"num_thread", "1"},
		"num_keep 1":                   {"num_keep", "1"},
		"seed 1":                       {"seed", "1"},
		"num_predict 1":                {"num_predict", "1"},
		"top_k 1":                      {"top_k", "1"},
		"top_p 1.0":                    {"top_p", "1.0"},
		"min_p 0.05":                   {"min_p", "0.05"},
		"tfs_z 1.0":                    {"tfs_z", "1.0"},
		"typical_p 1.0":                {"typical_p", "1.0"},
		"repeat_last_n 1":              {"repeat_last_n", "1"},
		"temperature 1.0":              {"temperature", "1.0"},
		"repeat_penalty 1.0":           {"repeat_penalty", "1.0"},
		"presence_penalty 1.0":         {"presence_penalty", "1.0"},
		"frequency_penalty 1.0":        {"frequency_penalty", "1.0"},
		"mirostat 1":                   {"mirostat", "1"},
		"mirostat_tau 1.0":             {"mirostat_tau", "1.0"},
		"mirostat_eta 1.0":             {"mirostat_eta", "1.0"},
		"penalize_newline true":        {"penalize_newline", "true"},
		"stop ### User:":               {"stop", "### User:"},
		"stop ### User: ":              {"stop", "### User:"},
		"stop \"### User:\"":           {"stop", "### User:"},
		"stop \"### User: \"":          {"stop", "### User: "},
		"stop \"\"\"### User:\"\"\"":   {"stop", "### User:"},
		"stop \"\"\"### User:\n\"\"\"": {"stop", "### User:\n"},
		"stop <|endoftext|>":           {"stop", "<|endoftext|>"},
		"stop <|eot_id|>":              {"stop", "<|eot_id|>"},
		"stop </s>":                    {"stop", "</s>"},
	}

	for k, v := range cases {
		t.Run(k, func(t *testing.T) {
			var b bytes.Buffer
			fmt.Fprintln(&b, "FROM foo")
			fmt.Fprintln(&b, "PARAMETER", k)
			modelfile, err := ParseFile(&b)
			require.NoError(t, err)

			assert.Equal(t, []Command{
				{Name: "model", Args: "foo"},
				{Name: v.name, Args: v.value},
			}, modelfile.Commands)
		})
	}
}

func TestParseFileComments(t *testing.T) {
	cases := []struct {
		input    string
		expected []Command
	}{
		{
			`
# comment
FROM foo
	`,
			[]Command{
				{Name: "model", Args: "foo"},
			},
		},
	}

	for _, c := range cases {
		t.Run("", func(t *testing.T) {
			modelfile, err := ParseFile(strings.NewReader(c.input))
			require.NoError(t, err)
			assert.Equal(t, c.expected, modelfile.Commands)
		})
	}
}

func TestParseFileFormatParseFile(t *testing.T) {
	cases := []string{
		`
FROM foo
ADAPTER adapter1
LICENSE MIT
PARAMETER param1 value1
PARAMETER param2 value2
TEMPLATE template1
MESSAGE system You are a file parser. Always parse things.
MESSAGE user Hey there!
MESSAGE assistant Hello, I want to parse all the things!
`,
		`
FROM foo
ADAPTER adapter1
LICENSE MIT
PARAMETER param1 value1
PARAMETER param2 value2
TEMPLATE template1
MESSAGE system """
You are a store greeter. Always responsed with "Hello!".
"""
MESSAGE user Hey there!
MESSAGE assistant Hello, I want to parse all the things!
`,
		`
FROM foo
ADAPTER adapter1
LICENSE """
Very long and boring legal text.
Blah blah blah.
"Oh look, a quote!"
"""

PARAMETER param1 value1
PARAMETER param2 value2
TEMPLATE template1
MESSAGE system """
You are a store greeter. Always responsed with "Hello!".
"""
MESSAGE user Hey there!
MESSAGE assistant Hello, I want to parse all the things!
`,
		`
FROM foo
SYSTEM ""
`,
	}

	for _, c := range cases {
		t.Run("", func(t *testing.T) {
			modelfile, err := ParseFile(strings.NewReader(c))
			require.NoError(t, err)

			modelfile2, err := ParseFile(strings.NewReader(modelfile.String()))
			require.NoError(t, err)

			assert.Equal(t, modelfile, modelfile2)
		})
	}
}

func TestParseFileUTF16ParseFile(t *testing.T) {
	data := `FROM bob
PARAMETER param1 1
PARAMETER param2 4096
SYSTEM You are a utf16 file.
`

	expected := []Command{
		{Name: "model", Args: "bob"},
		{Name: "param1", Args: "1"},
		{Name: "param2", Args: "4096"},
		{Name: "system", Args: "You are a utf16 file."},
	}

	t.Run("le", func(t *testing.T) {
		var b bytes.Buffer
		require.NoError(t, binary.Write(&b, binary.LittleEndian, []byte{0xff, 0xfe}))
		require.NoError(t, binary.Write(&b, binary.LittleEndian, utf16.Encode([]rune(data))))

		actual, err := ParseFile(&b)
		require.NoError(t, err)

		assert.Equal(t, expected, actual.Commands)
	})

	t.Run("be", func(t *testing.T) {
		var b bytes.Buffer
		require.NoError(t, binary.Write(&b, binary.BigEndian, []byte{0xfe, 0xff}))
		require.NoError(t, binary.Write(&b, binary.BigEndian, utf16.Encode([]rune(data))))

		actual, err := ParseFile(&b)
		require.NoError(t, err)
		assert.Equal(t, expected, actual.Commands)
	})
}

func TestParseMultiByte(t *testing.T) {
	input := `FROM test
	SYSTEM 你好👋`

	expect := []Command{
		{Name: "model", Args: "test"},
		{Name: "system", Args: "你好👋"},
	}

	encodings := []encoding.Encoding{
		unicode.UTF8,
		unicode.UTF16(unicode.LittleEndian, unicode.UseBOM),
		unicode.UTF16(unicode.BigEndian, unicode.UseBOM),
	}

	for _, encoding := range encodings {
		t.Run(fmt.Sprintf("%s", encoding), func(t *testing.T) {
			s, err := encoding.NewEncoder().String(input)
			require.NoError(t, err)

			actual, err := ParseFile(strings.NewReader(s))
			require.NoError(t, err)

			assert.Equal(t, expect, actual.Commands)
		})
	}
}
