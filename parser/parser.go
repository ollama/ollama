package parser

import (
	"bufio"
	"fmt"
	"io"
	"strings"
)

type Command struct {
	Name string
	Arg  string
}

func Parse(reader io.Reader) ([]Command, error) {
	var commands []Command
	var foundModel bool

	scanner := bufio.NewScanner(reader)
	multiline := false
	var multilineCommand *Command
	for scanner.Scan() {
		line := scanner.Text()
		if multiline {
			// If we're in a multiline string and the line is """, end the multiline string.
			if strings.TrimSpace(line) == `"""` {
				multiline = false
				commands = append(commands, *multilineCommand)
			} else {
				// Otherwise, append the line to the multiline string.
				multilineCommand.Arg += "\n" + line
			}
			continue
		}
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}

		command := Command{}
		switch strings.ToUpper(fields[0]) {
		case "FROM":
			command.Name = "model"
			command.Arg = fields[1]
			if command.Arg == "" {
				return nil, fmt.Errorf("no model specified in FROM line")
			}
			foundModel = true
		case "PROMPT":
			command.Name = "prompt"
			if fields[1] == `"""` {
				multiline = true
				multilineCommand = &command
				multilineCommand.Arg = ""
			} else {
				command.Arg = strings.Join(fields[1:], " ")
			}
		case "PARAMETER":
			command.Name = fields[1]
			command.Arg = strings.Join(fields[2:], " ")
		default:
			continue
		}
		if !multiline {
			commands = append(commands, command)
		}
	}

	if !foundModel {
		return nil, fmt.Errorf("no FROM line for the model was specified")
	}

	if multiline {
		return nil, fmt.Errorf("unclosed multiline string")
	}
	return commands, scanner.Err()
}
