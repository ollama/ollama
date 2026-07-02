package tui

import (
	"strings"

	"github.com/mattn/go-runewidth"
)

func wrapText(text string, width int) []string {
	if width <= 0 {
		return strings.Split(text, "\n")
	}
	var lines []string
	for _, line := range strings.Split(text, "\n") {
		for runewidth.StringWidth(line) > width {
			cut := textDisplayWidthCut(line, width)
			lines = append(lines, line[:cut])
			line = line[cut:]
		}
		lines = append(lines, line)
	}
	return lines
}

func textDisplayWidthCut(line string, width int) int {
	if width <= 0 {
		return 0
	}
	used := 0
	for index, r := range line {
		w := runewidth.RuneWidth(r)
		if used+w > width {
			if index == 0 {
				return len(string(r))
			}
			return index
		}
		used += w
	}
	return len(line)
}
