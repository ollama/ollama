package parser

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_Parser(t *testing.T) {

	input := `
FROM model1
ADAPTER adapter1
LICENSE MIT
PARAMETER param1 value1
PARAMETER param2 value2
TEMPLATE template1
`

	reader := strings.NewReader(input)

	commands, err := Parse(reader)
	assert.Nil(t, err)

	expectedCommands := []Command{
		{Name: "model", Args: "model1"},
		{Name: "adapter", Args: "adapter1"},
		{Name: "license", Args: "MIT"},
		{Name: "param1", Args: "value1"},
		{Name: "param2", Args: "value2"},
		{Name: "template", Args: "template1"},
	}

	assert.Equal(t, expectedCommands, commands)
}

func Test_Parser_NoFromLine(t *testing.T) {

	input := `
PARAMETER param1 value1
PARAMETER param2 value2
`

	reader := strings.NewReader(input)

	_, err := Parse(reader)
	assert.ErrorContains(t, err, "no FROM line")
}

func Test_Parser_MissingValue(t *testing.T) {

	input := `
FROM foo
PARAMETER param1
`

	reader := strings.NewReader(input)

	_, err := Parse(reader)
	assert.ErrorContains(t, err, "missing value for [param1]")

}
