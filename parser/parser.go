package parser

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"strings"
)

const multilineString = `"""`

type Command struct {
	Name string
	Args string
}

func (c *Command) Reset() {
	c.Name = ""
	c.Args = ""
}

func Parse(reader io.Reader) ([]Command, error) {
	var commands []Command
	var command, modelCommand Command

	scanner := bufio.NewScanner(reader)
	scanner.Split(scanModelfile)
	for scanner.Scan() {
		line := scanner.Bytes()

		fields := bytes.SplitN(line, []byte(" "), 2)
		if len(fields) == 0 {
			continue
		}

		switch strings.ToUpper(string(fields[0])) {
		case "FROM":
			command.Name = "model"
			command.Args = string(fields[1])
			// copy command for validation
			modelCommand = command
		case "LICENSE", "TEMPLATE", "SYSTEM", "PROMPT":
			command.Name = strings.ToLower(string(fields[0]))
			command.Args = string(fields[1])
		case "PARAMETER":
			fields = bytes.SplitN(fields[1], []byte(" "), 2)
			command.Name = string(fields[0])
			command.Args = string(fields[1])
		default:
			return nil, fmt.Errorf("unknown command: %s", fields[0])
		}

		commands = append(commands, command)
		command.Reset()
	}

	if modelCommand.Args == "" {
		return nil, errors.New("no FROM line for the model was specified")
	}

	return commands, scanner.Err()
}

func scanModelfile(data []byte, atEOF bool) (advance int, token []byte, err error) {
	newline := bytes.IndexByte(data, '\n')

	if start := bytes.Index(data, []byte(multilineString)); start >= 0 && start < newline {
		end := bytes.Index(data[start+len(multilineString):], []byte(multilineString))
		if end < 0 {
			if atEOF {
				return 0, nil, errors.New("unterminated multiline string: " + multilineString)
			} else {
				return 0, nil, nil
			}
		}

		n := start + len(multilineString) + end + len(multilineString)
		return n, data[:n], nil
	}

	return bufio.ScanLines(data, atEOF)
}
