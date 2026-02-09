package config

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"golang.org/x/term"
)

// ANSI escape sequences for terminal formatting.
const (
	ansiHideCursor = "\033[?25l"
	ansiShowCursor = "\033[?25h"
	ansiBold       = "\033[1m"
	ansiReset      = "\033[0m"
	ansiGray       = "\033[37m"
	ansiGreen      = "\033[32m"
	ansiClearDown  = "\033[J"
)

const maxDisplayedItems = 10

// ErrCancelled is returned when the user cancels a selection.
var ErrCancelled = errors.New("cancelled")

// errCancelled is kept as an alias for backward compatibility within the package.
var errCancelled = ErrCancelled

type selectItem struct {
	Name        string
	Description string
	Recommended bool
}

type inputEvent int

const (
	eventNone inputEvent = iota
	eventEnter
	eventEscape
	eventUp
	eventDown
	eventTab
	eventBackspace
	eventChar
)

type selectState struct {
	items        []selectItem
	filter       string
	selected     int
	scrollOffset int
}

func newSelectState(items []selectItem) *selectState {
	return &selectState{items: items}
}

func (s *selectState) filtered() []selectItem {
	return filterItems(s.items, s.filter)
}

func (s *selectState) handleInput(event inputEvent, char byte) (done bool, result string, err error) {
	filtered := s.filtered()

	switch event {
	case eventEnter:
		if len(filtered) > 0 && s.selected < len(filtered) {
			return true, filtered[s.selected].Name, nil
		}
	case eventEscape:
		return true, "", errCancelled
	case eventBackspace:
		if len(s.filter) > 0 {
			s.filter = s.filter[:len(s.filter)-1]
			s.selected = 0
			s.scrollOffset = 0
		}
	case eventUp:
		if s.selected > 0 {
			s.selected--
			if s.selected < s.scrollOffset {
				s.scrollOffset = s.selected
			}
		}
	case eventDown:
		if s.selected < len(filtered)-1 {
			s.selected++
			if s.selected >= s.scrollOffset+maxDisplayedItems {
				s.scrollOffset = s.selected - maxDisplayedItems + 1
			}
		}
	case eventChar:
		s.filter += string(char)
		s.selected = 0
		s.scrollOffset = 0
	}

	return false, "", nil
}

type multiSelectState struct {
	items         []selectItem
	itemIndex     map[string]int
	filter        string
	highlighted   int
	scrollOffset  int
	checked       map[int]bool
	checkOrder    []int
	focusOnButton bool
}

func newMultiSelectState(items []selectItem, preChecked []string) *multiSelectState {
	s := &multiSelectState{
		items:     items,
		itemIndex: make(map[string]int, len(items)),
		checked:   make(map[int]bool),
	}

	for i, item := range items {
		s.itemIndex[item.Name] = i
	}

	for _, name := range preChecked {
		if idx, ok := s.itemIndex[name]; ok {
			s.checked[idx] = true
			s.checkOrder = append(s.checkOrder, idx)
		}
	}

	return s
}

func (s *multiSelectState) filtered() []selectItem {
	return filterItems(s.items, s.filter)
}

func (s *multiSelectState) toggleItem() {
	filtered := s.filtered()
	if len(filtered) == 0 || s.highlighted >= len(filtered) {
		return
	}

	item := filtered[s.highlighted]
	origIdx := s.itemIndex[item.Name]

	if s.checked[origIdx] {
		delete(s.checked, origIdx)
		for i, idx := range s.checkOrder {
			if idx == origIdx {
				s.checkOrder = append(s.checkOrder[:i], s.checkOrder[i+1:]...)
				break
			}
		}
	} else {
		s.checked[origIdx] = true
		s.checkOrder = append(s.checkOrder, origIdx)
	}
}

func (s *multiSelectState) handleInput(event inputEvent, char byte) (done bool, result []string, err error) {
	filtered := s.filtered()

	switch event {
	case eventEnter:
		if s.focusOnButton && len(s.checkOrder) > 0 {
			var res []string
			for _, idx := range s.checkOrder {
				res = append(res, s.items[idx].Name)
			}
			return true, res, nil
		} else if !s.focusOnButton {
			s.toggleItem()
		}
	case eventTab:
		if len(s.checkOrder) > 0 {
			s.focusOnButton = !s.focusOnButton
		}
	case eventEscape:
		return true, nil, errCancelled
	case eventBackspace:
		if len(s.filter) > 0 {
			s.filter = s.filter[:len(s.filter)-1]
			s.highlighted = 0
			s.scrollOffset = 0
			s.focusOnButton = false
		}
	case eventUp:
		if s.focusOnButton {
			s.focusOnButton = false
		} else if s.highlighted > 0 {
			s.highlighted--
			if s.highlighted < s.scrollOffset {
				s.scrollOffset = s.highlighted
			}
		}
	case eventDown:
		if s.focusOnButton {
			s.focusOnButton = false
		} else if s.highlighted < len(filtered)-1 {
			s.highlighted++
			if s.highlighted >= s.scrollOffset+maxDisplayedItems {
				s.scrollOffset = s.highlighted - maxDisplayedItems + 1
			}
		}
	case eventChar:
		s.filter += string(char)
		s.highlighted = 0
		s.scrollOffset = 0
		s.focusOnButton = false
	}

	return false, nil, nil
}

func (s *multiSelectState) selectedCount() int {
	return len(s.checkOrder)
}

// Terminal I/O handling

type terminalState struct {
	fd       int
	oldState *term.State
}

func enterRawMode() (*terminalState, error) {
	fd := int(os.Stdin.Fd())
	oldState, err := term.MakeRaw(fd)
	if err != nil {
		return nil, err
	}
	fmt.Fprint(os.Stderr, ansiHideCursor)
	return &terminalState{fd: fd, oldState: oldState}, nil
}

func (t *terminalState) restore() {
	fmt.Fprint(os.Stderr, ansiShowCursor)
	term.Restore(t.fd, t.oldState)
}

func clearLines(n int) {
	if n > 0 {
		fmt.Fprintf(os.Stderr, "\033[%dA", n)
		fmt.Fprint(os.Stderr, ansiClearDown)
	}
}

func parseInput(r io.Reader) (inputEvent, byte, error) {
	buf := make([]byte, 3)
	n, err := r.Read(buf)
	if err != nil {
		return 0, 0, err
	}

	switch {
	case n == 1 && buf[0] == 13:
		return eventEnter, 0, nil
	case n == 1 && (buf[0] == 3 || buf[0] == 27):
		return eventEscape, 0, nil
	case n == 1 && buf[0] == 9:
		return eventTab, 0, nil
	case n == 1 && buf[0] == 127:
		return eventBackspace, 0, nil
	case n == 3 && buf[0] == 27 && buf[1] == 91 && buf[2] == 65:
		return eventUp, 0, nil
	case n == 3 && buf[0] == 27 && buf[1] == 91 && buf[2] == 66:
		return eventDown, 0, nil
	case n == 1 && buf[0] >= 32 && buf[0] < 127:
		return eventChar, buf[0], nil
	}

	return eventNone, 0, nil
}

// Rendering

func renderSelect(w io.Writer, prompt string, s *selectState) int {
	filtered := s.filtered()

	if s.filter == "" {
		fmt.Fprintf(w, "%s %sType to filter...%s\r\n", prompt, ansiGray, ansiReset)
	} else {
		fmt.Fprintf(w, "%s %s\r\n", prompt, s.filter)
	}
	lineCount := 1

	if len(filtered) == 0 {
		fmt.Fprintf(w, "  %s(no matches)%s\r\n", ansiGray, ansiReset)
		lineCount++
	} else {
		displayCount := min(len(filtered), maxDisplayedItems)

		for i := range displayCount {
			idx := s.scrollOffset + i
			if idx >= len(filtered) {
				break
			}
			item := filtered[idx]
			prefix := "    "
			if idx == s.selected {
				prefix = "  " + ansiBold + "> "
			}
			if item.Description != "" {
				fmt.Fprintf(w, "%s%s%s %s- %s%s\r\n", prefix, item.Name, ansiReset, ansiGray, item.Description, ansiReset)
			} else {
				fmt.Fprintf(w, "%s%s%s\r\n", prefix, item.Name, ansiReset)
			}
			lineCount++
		}

		if remaining := len(filtered) - s.scrollOffset - displayCount; remaining > 0 {
			fmt.Fprintf(w, "  %s... and %d more%s\r\n", ansiGray, remaining, ansiReset)
			lineCount++
		}
	}

	return lineCount
}

func renderMultiSelect(w io.Writer, prompt string, s *multiSelectState) int {
	filtered := s.filtered()

	if s.filter == "" {
		fmt.Fprintf(w, "%s %sType to filter...%s\r\n", prompt, ansiGray, ansiReset)
	} else {
		fmt.Fprintf(w, "%s %s\r\n", prompt, s.filter)
	}
	lineCount := 1

	if len(filtered) == 0 {
		fmt.Fprintf(w, "  %s(no matches)%s\r\n", ansiGray, ansiReset)
		lineCount++
	} else {
		displayCount := min(len(filtered), maxDisplayedItems)

		for i := range displayCount {
			idx := s.scrollOffset + i
			if idx >= len(filtered) {
				break
			}
			item := filtered[idx]
			origIdx := s.itemIndex[item.Name]

			checkbox := "[ ]"
			if s.checked[origIdx] {
				checkbox = "[x]"
			}

			prefix := "  "
			suffix := ""
			if idx == s.highlighted && !s.focusOnButton {
				prefix = "> "
			}
			if len(s.checkOrder) > 0 && s.checkOrder[0] == origIdx {
				suffix = " " + ansiGray + "(default)" + ansiReset
			}

			desc := ""
			if item.Description != "" {
				desc = " " + ansiGray + "- " + item.Description + ansiReset
			}

			if idx == s.highlighted && !s.focusOnButton {
				fmt.Fprintf(w, "  %s%s %s %s%s%s%s\r\n", ansiBold, prefix, checkbox, item.Name, ansiReset, desc, suffix)
			} else {
				fmt.Fprintf(w, "  %s %s %s%s%s\r\n", prefix, checkbox, item.Name, desc, suffix)
			}
			lineCount++
		}

		if remaining := len(filtered) - s.scrollOffset - displayCount; remaining > 0 {
			fmt.Fprintf(w, "  %s... and %d more%s\r\n", ansiGray, remaining, ansiReset)
			lineCount++
		}
	}

	fmt.Fprintf(w, "\r\n")
	lineCount++
	count := s.selectedCount()
	switch {
	case count == 0:
		fmt.Fprintf(w, "  %sSelect at least one model.%s\r\n", ansiGray, ansiReset)
	case s.focusOnButton:
		fmt.Fprintf(w, "  %s> [ Continue ]%s %s(%d selected)%s\r\n", ansiBold, ansiReset, ansiGray, count, ansiReset)
	default:
		fmt.Fprintf(w, "    %s[ Continue ] (%d selected) - press Tab%s\r\n", ansiGray, count, ansiReset)
	}
	lineCount++

	return lineCount
}

// selectPrompt prompts the user to select a single item from a list.
func selectPrompt(prompt string, items []selectItem) (string, error) {
	if len(items) == 0 {
		return "", fmt.Errorf("no items to select from")
	}

	ts, err := enterRawMode()
	if err != nil {
		return "", err
	}
	defer ts.restore()

	state := newSelectState(items)
	var lastLineCount int

	render := func() {
		clearLines(lastLineCount)
		lastLineCount = renderSelect(os.Stderr, prompt, state)
	}

	render()

	for {
		event, char, err := parseInput(os.Stdin)
		if err != nil {
			return "", err
		}

		done, result, err := state.handleInput(event, char)
		if done {
			clearLines(lastLineCount)
			if err != nil {
				return "", err
			}
			return result, nil
		}

		render()
	}
}

// multiSelectPrompt prompts the user to select multiple items from a list.
func multiSelectPrompt(prompt string, items []selectItem, preChecked []string) ([]string, error) {
	if len(items) == 0 {
		return nil, fmt.Errorf("no items to select from")
	}

	ts, err := enterRawMode()
	if err != nil {
		return nil, err
	}
	defer ts.restore()

	state := newMultiSelectState(items, preChecked)
	var lastLineCount int

	render := func() {
		clearLines(lastLineCount)
		lastLineCount = renderMultiSelect(os.Stderr, prompt, state)
	}

	render()

	for {
		event, char, err := parseInput(os.Stdin)
		if err != nil {
			return nil, err
		}

		done, result, err := state.handleInput(event, char)
		if done {
			clearLines(lastLineCount)
			if err != nil {
				return nil, err
			}
			return result, nil
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

	fmt.Fprintf(os.Stderr, "%s (\033[1my\033[0m/n) ", prompt)

	buf := make([]byte, 1)
	for {
		if _, err := os.Stdin.Read(buf); err != nil {
			return false, err
		}

		switch buf[0] {
		case 'Y', 'y', 13:
			fmt.Fprintf(os.Stderr, "yes\r\n")
			return true, nil
		case 'N', 'n', 27, 3:
			fmt.Fprintf(os.Stderr, "no\r\n")
			return false, nil
		}
	}
}

func filterItems(items []selectItem, filter string) []selectItem {
	if filter == "" {
		return items
	}
	var result []selectItem
	filterLower := strings.ToLower(filter)
	for _, item := range items {
		if strings.Contains(strings.ToLower(item.Name), filterLower) {
			result = append(result, item)
		}
	}
	return result
}
