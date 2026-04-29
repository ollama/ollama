// Package agent provides agent loop orchestration and tool approval.
package agent

import (
	"fmt"
	"os"
	"strings"
	"sync"

	"golang.org/x/term"
)

// ApprovalDecision represents the user's decision for a tool execution.
type ApprovalDecision int

const (
	// ApprovalDeny means the user denied execution.
	ApprovalDeny ApprovalDecision = iota
	// ApprovalOnce means execute this one time only.
	ApprovalOnce
	// ApprovalAlways means add to session allowlist.
	ApprovalAlways
)

// ApprovalResult contains the decision and optional deny reason.
type ApprovalResult struct {
	Decision   ApprovalDecision
	DenyReason string
}

// Option labels for the selector (numbered for quick selection)
var optionLabels = []string{
	"1. Execute once",
	"2. Allow for this session",
	"3. Deny",
}

// ApprovalPromptOptions controls warning text and allowlist context shown in the selector.
type ApprovalPromptOptions struct {
	Warnings      []string
	AllowlistInfo string
}

// ToolDisplayName returns the human-readable display name for a tool.
func ToolDisplayName(toolName string) string {
	if displayName, ok := bashToolDisplayName(toolName); ok {
		return displayName
	}
	if displayName, ok := browserToolDisplayName(toolName); ok {
		return displayName
	}
	// Default: capitalize first letter and replace underscores with spaces
	name := strings.ReplaceAll(toolName, "_", " ")
	if len(name) > 0 {
		return strings.ToUpper(name[:1]) + name[1:]
	}
	return toolName
}

// ApprovalManager manages tool execution approvals.
type ApprovalManager struct {
	allowlist map[string]bool // exact matches
	prefixes  map[string]bool // prefix matches for bash commands (e.g., "cat:tools/")
	mu        sync.RWMutex
}

// NewApprovalManager creates a new approval manager.
func NewApprovalManager() *ApprovalManager {
	return &ApprovalManager{
		allowlist: make(map[string]bool),
		prefixes:  make(map[string]bool),
	}
}

// AllowlistKey generates the key for exact allowlist lookup.
func AllowlistKey(toolName string, args map[string]any) string {
	if toolName == "bash" {
		if key, ok := bashAllowlistKey(args); ok {
			return key
		}
	}
	return toolName
}

// IsAllowed checks if a tool/command is allowed (exact match or prefix match).
// For bash commands, hierarchical path matching is used - if "cat:tools/" is allowed,
// then "cat:tools/subdir/" is also allowed (subdirectories inherit parent permissions).
func (a *ApprovalManager) IsAllowed(toolName string, args map[string]any) bool {
	a.mu.RLock()
	defer a.mu.RUnlock()

	// Check exact match first
	key := AllowlistKey(toolName, args)
	if a.allowlist[key] {
		return true
	}

	if toolName == "bash" {
		return a.isAllowedBash(args)
	}

	return a.allowlist[toolName]
}

// AddToAllowlist adds a tool/command to the session allowlist.
// For bash commands, it adds the prefix pattern instead of exact command.
func (a *ApprovalManager) AddToAllowlist(toolName string, args map[string]any) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if toolName == "bash" && a.addToAllowlistBash(args) {
		return
	}
	a.allowlist[toolName] = true
}

func withDefaultBashApprovalOptions(args map[string]any, options ApprovalPromptOptions) ApprovalPromptOptions {
	cmd, ok := args["command"].(string)
	if !ok {
		return options
	}

	defaults := BuildBashApprovalOptions(cmd)
	if options.AllowlistInfo == "" {
		options.AllowlistInfo = defaults.AllowlistInfo
	}
	if len(options.Warnings) == 0 {
		options.Warnings = defaults.Warnings
		return options
	}

	for _, warning := range defaults.Warnings {
		options.Warnings = appendUniqueWarning(options.Warnings, warning)
	}

	return options
}

func appendUniqueWarning(warnings []string, warning string) []string {
	for _, existing := range warnings {
		if existing == warning {
			return warnings
		}
	}
	return append(warnings, warning)
}

// RequestApproval prompts the user for approval to execute a tool.
// Returns the decision and optional deny reason.
func (a *ApprovalManager) RequestApproval(toolName string, args map[string]any, options ApprovalPromptOptions) (ApprovalResult, error) {
	// Format tool info for display
	toolDisplay := formatToolDisplay(toolName, args)
	if toolName == "bash" {
		options = withDefaultBashApprovalOptions(args, options)
	}

	// Enter raw mode for interactive selection
	fd := int(os.Stdin.Fd())
	oldState, err := term.MakeRaw(fd)
	if err != nil {
		// Fallback to simple input if terminal control fails
		return a.fallbackApproval(toolDisplay, options.Warnings)
	}

	// Flush any pending stdin input before starting selector
	// This prevents buffered input from causing double-press issues
	flushStdin(fd)

	// Run interactive selector
	selected, denyReason, err := runSelector(fd, oldState, toolDisplay, options.Warnings, options.AllowlistInfo)
	if err != nil {
		term.Restore(fd, oldState)
		return ApprovalResult{Decision: ApprovalDeny}, err
	}

	// Restore terminal
	term.Restore(fd, oldState)

	// Map selection to decision
	switch selected {
	case -1: // Ctrl+C cancelled
		return ApprovalResult{Decision: ApprovalDeny, DenyReason: "cancelled"}, nil
	case 0:
		return ApprovalResult{Decision: ApprovalOnce}, nil
	case 1:
		return ApprovalResult{Decision: ApprovalAlways}, nil
	default:
		return ApprovalResult{Decision: ApprovalDeny, DenyReason: denyReason}, nil
	}
}

// formatToolDisplay creates the display string for a tool call.
func formatToolDisplay(toolName string, args map[string]any) string {
	if toolName == "bash" {
		if display, ok := formatBashToolDisplay(args); ok {
			return display
		}
	}
	if display, ok := formatBrowserToolDisplay(toolName, args); ok {
		return display
	}

	var sb strings.Builder
	displayName := ToolDisplayName(toolName)
	sb.WriteString(fmt.Sprintf("Tool: %s", displayName))
	if len(args) > 0 {
		sb.WriteString("\nArguments: ")
		first := true
		for k, v := range args {
			if !first {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%s=%v", k, v))
			first = false
		}
	}
	return sb.String()
}

// selectorState holds the state for the interactive selector
type selectorState struct {
	toolDisplay   string
	selected      int
	totalLines    int
	termWidth     int
	termHeight    int
	boxWidth      int
	innerWidth    int
	denyReason    string
	editingDeny   bool
	warnings      []string
	allowlistInfo string // show what will be allowlisted (for "Allow for this session" option)
}

type selectorOutcome struct {
	done       bool
	selected   int
	denyReason string
}

// runSelector runs the interactive selector and returns the selected index and optional deny reason.
func runSelector(fd int, oldState *term.State, toolDisplay string, warnings []string, allowlistInfo string) (int, string, error) {
	state := &selectorState{
		toolDisplay:   toolDisplay,
		selected:      0,
		warnings:      warnings,
		allowlistInfo: allowlistInfo,
	}

	// Get terminal size
	state.termWidth, state.termHeight, _ = term.GetSize(fd)
	if state.termWidth < 20 {
		state.termWidth = 80 // fallback
	}

	// Calculate box width: 90% of terminal, min 24, max 60
	state.boxWidth = (state.termWidth * 90) / 100
	if state.boxWidth > 60 {
		state.boxWidth = 60
	}
	if state.boxWidth < 24 {
		state.boxWidth = 24
	}
	// Ensure box fits in terminal
	if state.boxWidth > state.termWidth-1 {
		state.boxWidth = state.termWidth - 1
	}
	state.innerWidth = state.boxWidth - 4 // account for "│ " and " │"

	// Calculate total lines (will be updated by render)
	state.totalLines = calculateTotalLines(state)

	// Hide cursor during selection (show when in deny mode)
	fmt.Fprint(os.Stderr, "\033[?25l")
	defer fmt.Fprint(os.Stderr, "\033[?25h") // Show cursor when done

	// Initial render
	renderSelectorBox(state)

	for {
		// Read input
		buf := make([]byte, 8)
		n, err := os.Stdin.Read(buf)
		if err != nil {
			clearSelectorBox(state)
			return 2, "", err
		}

		outcome, changed := handleSelectorInput(state, buf[:n])
		if outcome.done {
			clearSelectorBox(state)
			return outcome.selected, outcome.denyReason, nil
		}
		if changed {
			redrawSelector(state)
		}
	}
}

// wrapText wraps text to fit within maxWidth, returning lines
func wrapText(text string, maxWidth int) []string {
	if maxWidth < 5 {
		maxWidth = 5
	}
	var lines []string
	for _, line := range strings.Split(text, "\n") {
		if len(line) <= maxWidth {
			lines = append(lines, line)
			continue
		}
		// Wrap long lines
		for len(line) > maxWidth {
			// Try to break at space
			breakAt := maxWidth
			for i := maxWidth; i > maxWidth/2; i-- {
				if i < len(line) && line[i] == ' ' {
					breakAt = i
					break
				}
			}
			lines = append(lines, line[:breakAt])
			line = strings.TrimLeft(line[breakAt:], " ")
		}
		if len(line) > 0 {
			lines = append(lines, line)
		}
	}
	return lines
}

func denyReasonWrapWidth(state *selectorState) int {
	width := state.innerWidth - len(denyReasonPrefixText(state))
	if width < 10 {
		return 10
	}
	return width
}

func denyReasonPrefixText(state *selectorState) string {
	if state.editingDeny || state.denyReason != "" {
		return "3. Deny: "
	}
	return "3. Deny (tab to edit)"
}

func denyReasonPrefixDisplay(state *selectorState) string {
	if state.editingDeny || state.denyReason != "" {
		return "3. Deny: "
	}
	return "3. Deny \033[90m(tab to edit)\033[0m"
}

func getDenyReasonLines(state *selectorState) []string {
	if state.denyReason == "" {
		return nil
	}
	return wrapText(state.denyReason, denyReasonWrapWidth(state))
}

func getHintText(state *selectorState) string {
	parts := []string{
		"up/down select",
		"enter confirm",
		"1-3 quick select",
		"ctrl+c cancel",
	}
	return strings.Join(parts, ", ")
}

func displayWrapWidth(state *selectorState) int {
	if state.termWidth <= 11 {
		return 10
	}
	return state.termWidth - 1
}

func warningContentWrapWidth(state *selectorState) int {
	width := displayWrapWidth(state) - len("warning: ")
	if width < 10 {
		return 10
	}
	return width
}

func getWrappedToolLines(state *selectorState) []string {
	var lines []string
	for _, line := range strings.Split(state.toolDisplay, "\n") {
		lines = append(lines, wrapText(line, displayWrapWidth(state))...)
	}
	return lines
}

func getWarningContentLines(state *selectorState, warning string) []string {
	return wrapText(warning, warningContentWrapWidth(state))
}

func warningLineCount(state *selectorState) int {
	if len(state.warnings) == 0 {
		return 0
	}

	total := 1
	for _, warning := range state.warnings {
		total += len(getWarningContentLines(state, warning))
	}
	return total
}

// getHintLines returns the hint text wrapped to terminal width
func getHintLines(state *selectorState) []string {
	hint := getHintText(state)
	if state.termWidth >= len(hint)+1 {
		return []string{hint}
	}
	// Wrap hint to multiple lines
	return wrapText(hint, state.termWidth-1)
}

// calculateTotalLines calculates how many lines the selector will use
func calculateTotalLines(state *selectorState) int {
	toolLines := getWrappedToolLines(state)
	hintLines := getHintLines(state)
	warningLines := warningLineCount(state)
	optionLines := len(optionLabels) - 1 + max(1, len(getDenyReasonLines(state)))
	return warningLines + len(toolLines) + 1 + optionLines + 1 + len(hintLines)
}

// renderSelectorBox renders the selector (minimal, no box)
func renderSelectorBox(state *selectorState) {
	toolLines := getWrappedToolLines(state)
	hintLines := getHintLines(state)

	if len(state.warnings) > 0 {
		for _, warning := range state.warnings {
			lines := getWarningContentLines(state, warning)
			for i, line := range lines {
				if i == 0 {
					fmt.Fprintf(os.Stderr, "\033[31;1mwarning:\033[0m %s\033[K\r\n", line)
					continue
				}
				fmt.Fprintf(os.Stderr, "%s%s\033[K\r\n", strings.Repeat(" ", len("warning: ")), line)
			}
		}
		fmt.Fprintf(os.Stderr, "\033[K\r\n") // blank line after warning
	}

	// Draw tool info (plain white)
	for _, line := range toolLines {
		fmt.Fprintf(os.Stderr, "%s\033[K\r\n", line)
	}

	// Blank line separator
	fmt.Fprintf(os.Stderr, "\033[K\r\n")

	for i, label := range optionLabels {
		if i == 2 {
			renderDenyOption(state)
		} else {
			displayLabel := label
			if i == 1 && state.allowlistInfo != "" {
				displayLabel = fmt.Sprintf("%s  \033[90m%s\033[0m", label, state.allowlistInfo)
			}
			if i == state.selected {
				fmt.Fprintf(os.Stderr, "  \033[1m%s\033[0m\033[K\r\n", displayLabel)
			} else {
				fmt.Fprintf(os.Stderr, "  \033[37m%s\033[0m\033[K\r\n", displayLabel)
			}
		}
	}

	// Blank line before hint
	fmt.Fprintf(os.Stderr, "\033[K\r\n")

	// Draw hint (dark grey)
	for i, line := range hintLines {
		if i == len(hintLines)-1 {
			fmt.Fprintf(os.Stderr, "\033[90m%s\033[0m\033[K", line)
		} else {
			fmt.Fprintf(os.Stderr, "\033[90m%s\033[0m\033[K\r\n", line)
		}
	}
}

func renderDenyOption(state *selectorState) {
	labelStyle := "\033[37m"
	if state.selected == 2 {
		labelStyle = "\033[1m"
	}
	prefixText := denyReasonPrefixText(state)
	prefixDisplay := denyReasonPrefixDisplay(state)

	if state.denyReason == "" {
		fmt.Fprintf(os.Stderr, "  %s%s\033[0m\033[K\r\n", labelStyle, prefixDisplay)
		return
	}

	lines := getDenyReasonLines(state)
	indent := strings.Repeat(" ", len(prefixText))
	for i, line := range lines {
		prefix := indent
		if i == 0 {
			prefix = prefixDisplay
		}
		fmt.Fprintf(os.Stderr, "  %s%s\033[0m%s\033[K\r\n", labelStyle, prefix, line)
	}
}

func redrawSelector(state *selectorState) {
	nextTotalLines := calculateTotalLines(state)
	clearSelectorLines(selectorClearLineCount(state.totalLines, nextTotalLines))
	state.totalLines = nextTotalLines
	renderSelectorBox(state)
}

func handleSelectorInput(state *selectorState, input []byte) (selectorOutcome, bool) {
	changed := false

	for i := 0; i < len(input); i++ {
		ch := input[i]

		if ch == 27 && i+2 < len(input) && input[i+1] == '[' {
			outcome, didChange := handleSelectorArrow(state, input[i+2])
			changed = changed || didChange
			i += 2
			if outcome.done {
				return outcome, changed
			}
			continue
		}

		outcome, didChange := handleSelectorByte(state, ch)
		changed = changed || didChange
		if outcome.done {
			return outcome, changed
		}
	}

	return selectorOutcome{}, changed
}

func handleSelectorArrow(state *selectorState, arrow byte) (selectorOutcome, bool) {
	switch arrow {
	case 'A':
		if state.editingDeny {
			clearDenyEdit(state)
			if state.selected > 0 {
				state.selected--
			}
			return selectorOutcome{}, true
		}
		if state.selected > 0 {
			state.selected--
			return selectorOutcome{}, true
		}
	case 'B':
		if state.editingDeny {
			clearDenyEdit(state)
			return selectorOutcome{}, true
		}
		if state.selected < len(optionLabels)-1 {
			state.selected++
			return selectorOutcome{}, true
		}
	}

	return selectorOutcome{}, false
}

func handleSelectorByte(state *selectorState, ch byte) (selectorOutcome, bool) {
	switch {
	case ch == 3:
		return selectorOutcome{done: true, selected: -1}, false
	case ch == 13:
		if state.selected == 2 {
			return selectorOutcome{done: true, selected: 2, denyReason: state.denyReason}, false
		}
		return selectorOutcome{done: true, selected: state.selected}, false
	case ch == '\t':
		return selectorOutcome{}, enterDenyEdit(state)
	case ch == 127 || ch == 8:
		if len(state.denyReason) == 0 {
			return selectorOutcome{}, false
		}
		enterDenyEdit(state)
		runes := []rune(state.denyReason)
		state.denyReason = string(runes[:len(runes)-1])
		return selectorOutcome{}, true
	case ch == 27:
		if state.editingDeny || state.denyReason != "" {
			clearDenyEdit(state)
			return selectorOutcome{}, true
		}
		return selectorOutcome{}, false
	case state.editingDeny:
		if isPrintableASCII(ch) {
			return selectorOutcome{}, appendDenyReason(state, ch)
		}
	case ch >= '1' && ch <= '3':
		selected := int(ch - '1')
		if selected == 2 {
			return selectorOutcome{done: true, selected: 2, denyReason: state.denyReason}, false
		}
		return selectorOutcome{done: true, selected: selected}, false
	case isPrintableASCII(ch):
		enterDenyEdit(state)
		return selectorOutcome{}, appendDenyReason(state, ch)
	}

	return selectorOutcome{}, false
}

func enterDenyEdit(state *selectorState) bool {
	changed := false
	if state.selected != 2 {
		state.selected = 2
		changed = true
	}
	if !state.editingDeny {
		state.editingDeny = true
		changed = true
	}
	return changed
}

func clearDenyEdit(state *selectorState) {
	state.editingDeny = false
	state.denyReason = ""
}

func appendDenyReason(state *selectorState, ch byte) bool {
	if !isPrintableASCII(ch) {
		return false
	}

	enterDenyEdit(state)
	maxLen := max(denyReasonWrapWidth(state)*4, 64)
	if len([]rune(state.denyReason)) >= maxLen {
		return false
	}
	state.denyReason += string(ch)
	return true
}

func isPrintableASCII(ch byte) bool {
	return ch >= 32 && ch < 127
}

// clearSelectorBox clears the selector from screen
func clearSelectorBox(state *selectorState) {
	clearSelectorLines(state.totalLines)
}

func selectorClearLineCount(previousTotalLines int, nextTotalLines int) int {
	return max(previousTotalLines, nextTotalLines)
}

func clearSelectorLines(totalLines int) {
	if totalLines <= 0 {
		return
	}
	// Clear the current line (hint line) first
	fmt.Fprint(os.Stderr, "\r\033[K")
	// Move up and clear each remaining line
	for range totalLines - 1 {
		fmt.Fprint(os.Stderr, "\033[A\033[K")
	}
	fmt.Fprint(os.Stderr, "\r")
}

// fallbackApproval handles approval when terminal control isn't available.
func (a *ApprovalManager) fallbackApproval(toolDisplay string, warnings []string) (ApprovalResult, error) {
	fmt.Fprintln(os.Stderr)
	for _, warning := range warnings {
		fmt.Fprintf(os.Stderr, "\033[31;1mwarning:\033[0m %s\n", warning)
	}
	if len(warnings) > 0 {
		fmt.Fprintln(os.Stderr)
	}
	fmt.Fprintln(os.Stderr, toolDisplay)
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr, "[1] Execute once  [2] Allow for this session  [3] Deny")
	fmt.Fprint(os.Stderr, "choice: ")

	var input string
	fmt.Scanln(&input)

	switch input {
	case "1":
		return ApprovalResult{Decision: ApprovalOnce}, nil
	case "2":
		return ApprovalResult{Decision: ApprovalAlways}, nil
	default:
		fmt.Fprint(os.Stderr, "Reason (optional): ")
		var reason string
		fmt.Scanln(&reason)
		return ApprovalResult{Decision: ApprovalDeny, DenyReason: reason}, nil
	}
}

// Reset clears the session allowlist.
func (a *ApprovalManager) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.allowlist = make(map[string]bool)
	a.prefixes = make(map[string]bool)
}

// AllowedTools returns a list of tools and prefixes in the allowlist.
func (a *ApprovalManager) AllowedTools() []string {
	a.mu.RLock()
	defer a.mu.RUnlock()

	tools := make([]string, 0, len(a.allowlist)+len(a.prefixes))
	for tool := range a.allowlist {
		tools = append(tools, tool)
	}
	for prefix := range a.prefixes {
		tools = append(tools, prefix+"*")
	}
	return tools
}

// FormatApprovalResult returns a formatted string showing the approval result.
func FormatApprovalResult(toolName string, args map[string]any, result ApprovalResult) string {
	var label string

	switch result.Decision {
	case ApprovalOnce:
		label = "Approved"
	case ApprovalAlways:
		label = "Always allowed"
	case ApprovalDeny:
		label = "Denied"
	}

	if toolName == "bash" {
		if formatted, ok := formatBashApprovalResult(label, args); ok {
			return formatted
		}
	}
	if formatted, ok := formatBrowserApprovalResult(label, toolName, args); ok {
		return formatted
	}

	return fmt.Sprintf("\033[1m%s:\033[0m %s", label, ToolDisplayName(toolName))
}

// FormatDenyResult returns the tool result message when a tool is denied.
func FormatDenyResult(toolName string, reason string) string {
	if reason != "" {
		return fmt.Sprintf("User denied execution of %s. Reason: %s", toolName, reason)
	}
	return fmt.Sprintf("User denied execution of %s.", toolName)
}

func truncateDisplayText(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	if limit <= 3 {
		return value[:limit]
	}
	return value[:limit-3] + "..."
}

// PromptYesNo displays a simple Yes/No prompt and returns the user's choice.
// Returns true for Yes, false for No.
func PromptYesNo(question string) (bool, error) {
	fd := int(os.Stdin.Fd())
	oldState, err := term.MakeRaw(fd)
	if err != nil {
		return false, err
	}
	defer term.Restore(fd, oldState)

	selected := 0 // 0 = Yes, 1 = No
	options := []string{"Yes", "No"}

	// Hide cursor
	fmt.Fprint(os.Stderr, "\033[?25l")
	defer fmt.Fprint(os.Stderr, "\033[?25h")

	renderYesNo := func() {
		// Move to start of line and clear
		fmt.Fprintf(os.Stderr, "\r\033[K")
		fmt.Fprintf(os.Stderr, "%s  ", question)
		for i, opt := range options {
			if i == selected {
				fmt.Fprintf(os.Stderr, "\033[1m%s\033[0m  ", opt)
			} else {
				fmt.Fprintf(os.Stderr, "\033[37m%s\033[0m  ", opt)
			}
		}
	}

	renderYesNo()

	buf := make([]byte, 3)
	for {
		n, err := os.Stdin.Read(buf)
		if err != nil {
			return false, err
		}

		if n == 1 {
			switch buf[0] {
			case 'y', 'Y':
				selected = 0
				renderYesNo()
			case 'n', 'N':
				selected = 1
				renderYesNo()
			case '\r', '\n': // Enter
				fmt.Fprintf(os.Stderr, "\r\033[K") // Clear line
				return selected == 0, nil
			case 3: // Ctrl+C
				fmt.Fprintf(os.Stderr, "\r\033[K")
				return false, nil
			case 27: // Escape - could be arrow key
				// Read more bytes for arrow keys
				continue
			}
		} else if n == 3 && buf[0] == 27 && buf[1] == 91 {
			// Arrow keys
			switch buf[2] {
			case 'D': // Left
				if selected > 0 {
					selected--
				}
				renderYesNo()
			case 'C': // Right
				if selected < len(options)-1 {
					selected++
				}
				renderYesNo()
			}
		}
	}
}
