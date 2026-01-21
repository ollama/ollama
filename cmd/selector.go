package cmd

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"golang.org/x/term"
)

const maxDisplayedItems = 10

var ErrCancelled = errors.New("cancelled")

type SelectItem struct {
	Name        string
	Description string
}

// terminalState manages raw terminal mode for interactive selection
type terminalState struct {
	fd       int
	oldState *term.State
}

// enterRawMode puts terminal in raw mode with cursor hidden
func enterRawMode() (*terminalState, error) {
	fd := int(os.Stdin.Fd())
	oldState, err := term.MakeRaw(fd)
	if err != nil {
		return nil, err
	}
	fmt.Fprint(os.Stderr, "\033[?25l") // hide cursor
	return &terminalState{fd: fd, oldState: oldState}, nil
}

// restore restores terminal state and shows cursor
func (t *terminalState) restore() {
	fmt.Fprint(os.Stderr, "\033[?25h") // show cursor
	term.Restore(t.fd, t.oldState)
}

// clearLines moves cursor up n lines and clears from there
func clearLines(n int) {
	if n > 0 {
		fmt.Fprintf(os.Stderr, "\033[%dA", n)
		fmt.Fprint(os.Stderr, "\033[J")
	}
}

func Select(prompt string, items []SelectItem) (string, error) {
	if len(items) == 0 {
		return "", fmt.Errorf("no items to select from")
	}

	ts, err := enterRawMode()
	if err != nil {
		return "", err
	}
	defer ts.restore()

	var filter string
	selected := 0
	scrollOffset := 0
	var lastLineCount int

	render := func() {
		filtered := filterItems(items, filter)
		clearLines(lastLineCount)

		fmt.Fprintf(os.Stderr, "%s %s\r\n", prompt, filter)
		lineCount := 1

		if len(filtered) == 0 {
			fmt.Fprintf(os.Stderr, "  \033[37m(no matches)\033[0m\r\n")
			lineCount++
		} else {
			displayCount := min(len(filtered), maxDisplayedItems)

			for i := range displayCount {
				idx := scrollOffset + i
				if idx >= len(filtered) {
					break
				}
				item := filtered[idx]
				prefix := "    "
				if idx == selected {
					prefix = "  \033[1m> "
				}
				if item.Description != "" {
					fmt.Fprintf(os.Stderr, "%s%s\033[0m \033[37m- %s\033[0m\r\n", prefix, item.Name, item.Description)
				} else {
					fmt.Fprintf(os.Stderr, "%s%s\033[0m\r\n", prefix, item.Name)
				}
				lineCount++
			}

			if remaining := len(filtered) - scrollOffset - displayCount; remaining > 0 {
				fmt.Fprintf(os.Stderr, "  \033[37m... and %d more\033[0m\r\n", remaining)
				lineCount++
			}
		}

		lastLineCount = lineCount
	}

	render()

	buf := make([]byte, 3)
	for {
		n, err := os.Stdin.Read(buf)
		if err != nil {
			return "", err
		}

		filtered := filterItems(items, filter)

		switch {
		case n == 1 && buf[0] == 13: // Enter
			if len(filtered) > 0 && selected < len(filtered) {
				clearLines(lastLineCount)
				return filtered[selected].Name, nil
			}
		case n == 1 && (buf[0] == 3 || buf[0] == 27): // Ctrl+C or Escape
			clearLines(lastLineCount)
			return "", ErrCancelled
		case n == 1 && buf[0] == 127: // Backspace
			if len(filter) > 0 {
				filter = filter[:len(filter)-1]
				selected = 0
				scrollOffset = 0
			}
		case n == 3 && buf[0] == 27 && buf[1] == 91: // Arrow keys
			if buf[2] == 65 && selected > 0 { // Up
				selected--
				if selected < scrollOffset {
					scrollOffset = selected
				}
			} else if buf[2] == 66 && selected < len(filtered)-1 { // Down
				selected++
				if selected >= scrollOffset+maxDisplayedItems {
					scrollOffset = selected - maxDisplayedItems + 1
				}
			}
		case n == 1 && buf[0] >= 32 && buf[0] < 127: // Printable chars
			filter += string(buf[0])
			selected = 0
			scrollOffset = 0
		}

		render()
	}
}

func filterItems(items []SelectItem, filter string) []SelectItem {
	if filter == "" {
		return items
	}
	var result []SelectItem
	filterLower := strings.ToLower(filter)
	for _, item := range items {
		if strings.Contains(strings.ToLower(item.Name), filterLower) {
			result = append(result, item)
		}
	}
	return result
}

func MultiSelect(prompt string, items []SelectItem, preChecked []string) ([]string, error) {
	if len(items) == 0 {
		return nil, fmt.Errorf("no items to select from")
	}

	ts, err := enterRawMode()
	if err != nil {
		return nil, err
	}
	defer ts.restore()

	var filter string
	highlighted := 0
	scrollOffset := 0
	checked := make(map[int]bool)
	var checkOrder []int
	var lastLineCount int
	focusOnButton := false

	// Build index lookup for O(1) access
	itemIndex := make(map[string]int, len(items))
	for i, item := range items {
		itemIndex[item.Name] = i
	}

	// Pre-check items in their original order (preserves default as first)
	for _, name := range preChecked {
		if idx, ok := itemIndex[name]; ok {
			checked[idx] = true
			checkOrder = append(checkOrder, idx)
		}
	}

	render := func() {
		filtered := filterItems(items, filter)
		clearLines(lastLineCount)

		fmt.Fprintf(os.Stderr, "%s %s\r\n", prompt, filter)
		lineCount := 1

		if len(filtered) == 0 {
			fmt.Fprintf(os.Stderr, "  \033[37m(no matches)\033[0m\r\n")
			lineCount++
		} else {
			displayCount := min(len(filtered), maxDisplayedItems)

			for i := range displayCount {
				idx := scrollOffset + i
				if idx >= len(filtered) {
					break
				}
				item := filtered[idx]
				origIdx := itemIndex[item.Name]

				checkbox := "[ ]"
				if checked[origIdx] {
					checkbox = "[x]"
				}

				prefix := "  "
				suffix := ""
				if idx == highlighted && !focusOnButton {
					prefix = "> "
				}
				if len(checkOrder) > 0 && checkOrder[0] == origIdx {
					suffix = " \033[37m(default)\033[0m"
				}

				if idx == highlighted && !focusOnButton {
					fmt.Fprintf(os.Stderr, "  \033[1m%s %s %s\033[0m%s\r\n", prefix, checkbox, item.Name, suffix)
				} else {
					fmt.Fprintf(os.Stderr, "  %s %s %s%s\r\n", prefix, checkbox, item.Name, suffix)
				}
				lineCount++
			}

			if remaining := len(filtered) - scrollOffset - displayCount; remaining > 0 {
				fmt.Fprintf(os.Stderr, "  \033[37m... and %d more\033[0m\r\n", remaining)
				lineCount++
			}
		}

		// Continue button
		fmt.Fprintf(os.Stderr, "\r\n")
		lineCount++
		count := len(checkOrder)
		switch {
		case count == 0:
			fmt.Fprintf(os.Stderr, "  \033[37mSelect at least one model.\033[0m\r\n")
		case focusOnButton:
			fmt.Fprintf(os.Stderr, "  \033[1m> [ Continue ]\033[0m \033[37m(%d selected)\033[0m\r\n", count)
		default:
			fmt.Fprintf(os.Stderr, "    \033[37m[ Continue ] (%d selected) - press Tab\033[0m\r\n", count)
		}
		lineCount++

		lastLineCount = lineCount
	}

	toggleItem := func() {
		filtered := filterItems(items, filter)
		if len(filtered) == 0 || highlighted >= len(filtered) {
			return
		}

		item := filtered[highlighted]
		origIdx := itemIndex[item.Name]

		if checked[origIdx] {
			delete(checked, origIdx)
			for i, idx := range checkOrder {
				if idx == origIdx {
					checkOrder = append(checkOrder[:i], checkOrder[i+1:]...)
					break
				}
			}
		} else {
			checked[origIdx] = true
			checkOrder = append(checkOrder, origIdx)
		}
	}

	render()

	buf := make([]byte, 3)
	for {
		n, err := os.Stdin.Read(buf)
		if err != nil {
			return nil, err
		}

		filtered := filterItems(items, filter)

		switch {
		case n == 1 && buf[0] == 13: // Enter
			if focusOnButton && len(checkOrder) > 0 {
				clearLines(lastLineCount)
				var result []string
				for _, idx := range checkOrder {
					result = append(result, items[idx].Name)
				}
				return result, nil
			} else if !focusOnButton {
				toggleItem()
			}
		case n == 1 && buf[0] == 9: // Tab
			if len(checkOrder) > 0 {
				focusOnButton = !focusOnButton
			}
		case n == 1 && (buf[0] == 3 || buf[0] == 27): // Ctrl+C or Escape
			clearLines(lastLineCount)
			return nil, ErrCancelled
		case n == 1 && buf[0] == 127: // Backspace
			if len(filter) > 0 {
				filter = filter[:len(filter)-1]
				highlighted = 0
				scrollOffset = 0
				focusOnButton = false
			}
		case n == 3 && buf[0] == 27 && buf[1] == 91: // Arrow keys
			if focusOnButton {
				// Any arrow key returns focus to list
				focusOnButton = false
			} else {
				if buf[2] == 65 && highlighted > 0 { // Up
					highlighted--
					if highlighted < scrollOffset {
						scrollOffset = highlighted
					}
				} else if buf[2] == 66 && highlighted < len(filtered)-1 { // Down
					highlighted++
					if highlighted >= scrollOffset+maxDisplayedItems {
						scrollOffset = highlighted - maxDisplayedItems + 1
					}
				}
			}
		case n == 1 && buf[0] >= 32 && buf[0] < 127: // Printable chars
			filter += string(buf[0])
			highlighted = 0
			scrollOffset = 0
			focusOnButton = false
		}

		render()
	}
}

func confirmPrompt(prompt string) (bool, error) {
	fd := int(os.Stdin.Fd())
	oldState, err := term.MakeRaw(fd)
	if err != nil {
		return false, err
	}
	defer term.Restore(fd, oldState)

	fmt.Fprintf(os.Stderr, "%s [y/n] ", prompt)

	buf := make([]byte, 1)
	for {
		if _, err := os.Stdin.Read(buf); err != nil {
			return false, err
		}

		switch buf[0] {
		case 'Y', 'y', 13: // Yes, Enter
			fmt.Fprintf(os.Stderr, "yes\r\n")
			return true, nil
		case 'N', 'n', 27, 3: // No, Escape, Ctrl+C
			fmt.Fprintf(os.Stderr, "no\r\n")
			return false, nil
		}
	}
}
