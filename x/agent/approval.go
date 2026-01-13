// Package agent provides agent loop orchestration and tool approval.
package agent

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
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

// toolDisplayNames maps internal tool names to human-readable display names.
var toolDisplayNames = map[string]string{
	"bash":       "Bash",
	"web_search": "Web Search",
	"web_fetch":  "Web Fetch",
}

// ToolDisplayName returns the human-readable display name for a tool.
func ToolDisplayName(toolName string) string {
	if displayName, ok := toolDisplayNames[toolName]; ok {
		return displayName
	}
	// Default: capitalize first letter and replace underscores with spaces
	name := strings.ReplaceAll(toolName, "_", " ")
	if len(name) > 0 {
		return strings.ToUpper(name[:1]) + name[1:]
	}
	return toolName
}

// autoAllowCommands are commands that are always allowed without prompting.
// These are zero-risk, read-only commands.
var autoAllowCommands = map[string]bool{
	"pwd":      true,
	"echo":     true,
	"date":     true,
	"whoami":   true,
	"hostname": true,
	"uname":    true,
}

// autoAllowPrefixes are command prefixes that are always allowed.
// These are read-only or commonly-needed development commands.
var autoAllowPrefixes = []string{
	// Git read-only
	"git status", "git log", "git diff", "git branch", "git show",
	"git remote -v", "git tag", "git stash list",
	// Package managers - run scripts
	"npm run", "npm test", "npm start",
	"bun run", "bun test",
	"uv run",
	"yarn run", "yarn test",
	"pnpm run", "pnpm test",
	// Package info
	"go list", "go version", "go env",
	"npm list", "npm ls", "npm version",
	"pip list", "pip show",
	"cargo tree", "cargo version",
	// Build commands
	"go build", "go test", "go fmt", "go vet",
	"make", "cmake",
	"cargo build", "cargo test", "cargo check",
}

// denyPatterns are dangerous command patterns that are always blocked.
var denyPatterns = []string{
	// Destructive commands
	"rm -rf", "rm -fr",
	"mkfs", "dd if=", "dd of=",
	"shred",
	"> /dev/", ">/dev/",
	// Privilege escalation
	"sudo ", "su ", "doas ",
	"chmod 777", "chmod -R 777",
	"chown ", "chgrp ",
	// Network exfiltration
	"curl -d", "curl --data", "curl -X POST", "curl -X PUT",
	"wget --post",
	"nc ", "netcat ",
	"scp ", "rsync ",
	// History and credentials
	"history",
	".bash_history", ".zsh_history",
	".ssh/id_rsa", ".ssh/id_dsa", ".ssh/id_ecdsa", ".ssh/id_ed25519",
	".ssh/config",
	".aws/credentials", ".aws/config",
	".gnupg/",
	"/etc/shadow", "/etc/passwd",
	// Dangerous patterns
	":(){ :|:& };:", // fork bomb
	"chmod +s",      // setuid
	"mkfifo",
}

// denyPathPatterns are file patterns that should never be accessed.
// These are checked as exact filename matches or path suffixes.
var denyPathPatterns = []string{
	".env",
	".env.local",
	".env.production",
	"credentials.json",
	"secrets.json",
	"secrets.yaml",
	"secrets.yml",
	".pem",
	".key",
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

// IsAutoAllowed checks if a bash command is auto-allowed (no prompt needed).
func IsAutoAllowed(command string) bool {
	command = strings.TrimSpace(command)

	// Check exact command match (first word)
	fields := strings.Fields(command)
	if len(fields) > 0 && autoAllowCommands[fields[0]] {
		return true
	}

	// Check prefix match
	for _, prefix := range autoAllowPrefixes {
		if strings.HasPrefix(command, prefix) {
			return true
		}
	}

	return false
}

// IsDenied checks if a bash command matches deny patterns.
// Returns true and the matched pattern if denied.
func IsDenied(command string) (bool, string) {
	commandLower := strings.ToLower(command)

	// Check deny patterns
	for _, pattern := range denyPatterns {
		if strings.Contains(commandLower, strings.ToLower(pattern)) {
			return true, pattern
		}
	}

	// Check deny path patterns
	for _, pattern := range denyPathPatterns {
		if strings.Contains(commandLower, strings.ToLower(pattern)) {
			return true, pattern
		}
	}

	return false, ""
}

// FormatDeniedResult returns the tool result message when a command is blocked.
func FormatDeniedResult(command string, pattern string) string {
	return fmt.Sprintf("Command blocked: this command matches a dangerous pattern (%s) and cannot be executed. If this command is necessary, please ask the user to run it manually.", pattern)
}

// extractBashPrefix extracts a prefix pattern from a bash command.
// For commands like "cat tools/tools_test.go | head -200", returns "cat:tools/"
// For commands without path args, returns empty string.
// Paths with ".." traversal that escape the base directory return empty string for security.
func extractBashPrefix(command string) string {
	// Split command by pipes and get the first part
	parts := strings.Split(command, "|")
	firstCmd := strings.TrimSpace(parts[0])

	// Split into command and args
	fields := strings.Fields(firstCmd)
	if len(fields) < 2 {
		return ""
	}

	baseCmd := fields[0]
	// Common commands that benefit from prefix allowlisting
	// These are typically safe for read operations on specific directories
	safeCommands := map[string]bool{
		"cat": true, "ls": true, "head": true, "tail": true,
		"less": true, "more": true, "file": true, "wc": true,
		"grep": true, "find": true, "tree": true, "stat": true,
		"sed": true,
	}

	if !safeCommands[baseCmd] {
		return ""
	}

	// Find the first path-like argument (must contain / or \ or start with .)
	// First pass: look for clear paths (containing path separators or starting with .)
	for _, arg := range fields[1:] {
		// Skip flags
		if strings.HasPrefix(arg, "-") {
			continue
		}
		// Skip numeric arguments (e.g., "head -n 100")
		if isNumeric(arg) {
			continue
		}
		// Only process if it looks like a path (contains / or \ or starts with .)
		if !strings.Contains(arg, "/") && !strings.Contains(arg, "\\") && !strings.HasPrefix(arg, ".") {
			continue
		}
		// Normalize to forward slashes for consistent cross-platform matching
		arg = strings.ReplaceAll(arg, "\\", "/")

		// Security: reject absolute paths
		if path.IsAbs(arg) {
			return "" // Absolute path - don't create prefix
		}

		// Normalize the path using stdlib path.Clean (resolves . and ..)
		cleaned := path.Clean(arg)

		// Security: reject if cleaned path escapes to parent directory
		if strings.HasPrefix(cleaned, "..") {
			return "" // Path escapes - don't create prefix
		}

		// Security: if original had "..", verify cleaned path didn't escape to sibling
		// e.g., "tools/a/b/../../../etc" -> "etc" (escaped tools/ to sibling)
		if strings.Contains(arg, "..") {
			origBase := strings.SplitN(arg, "/", 2)[0]
			cleanedBase := strings.SplitN(cleaned, "/", 2)[0]
			if origBase != cleanedBase {
				return "" // Path escaped to sibling directory
			}
		}

		// Check if arg ends with / (explicit directory)
		isDir := strings.HasSuffix(arg, "/")

		// Get the directory part
		var dir string
		if isDir {
			dir = cleaned
		} else {
			dir = path.Dir(cleaned)
		}

		if dir == "." {
			return fmt.Sprintf("%s:./", baseCmd)
		}
		return fmt.Sprintf("%s:%s/", baseCmd, dir)
	}

	// Second pass: if no clear path found, use the first non-flag argument as a filename
	for _, arg := range fields[1:] {
		if strings.HasPrefix(arg, "-") {
			continue
		}
		if isNumeric(arg) {
			continue
		}
		// Treat as filename in current dir
		return fmt.Sprintf("%s:./", baseCmd)
	}

	return ""
}

// isNumeric checks if a string is a numeric value
func isNumeric(s string) bool {
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return len(s) > 0
}

// isCommandOutsideCwd checks if a bash command targets paths outside the current working directory.
// Returns true if any path argument would access files outside cwd.
func isCommandOutsideCwd(command string) bool {
	cwd, err := os.Getwd()
	if err != nil {
		return false // Can't determine, assume safe
	}

	// Split command by pipes and semicolons to check all parts
	parts := strings.FieldsFunc(command, func(r rune) bool {
		return r == '|' || r == ';' || r == '&'
	})

	for _, part := range parts {
		part = strings.TrimSpace(part)
		fields := strings.Fields(part)
		if len(fields) == 0 {
			continue
		}

		// Check each argument that looks like a path
		for _, arg := range fields[1:] {
			// Skip flags
			if strings.HasPrefix(arg, "-") {
				continue
			}

			// Treat POSIX-style absolute paths as outside cwd on all platforms.
			if strings.HasPrefix(arg, "/") || strings.HasPrefix(arg, "\\") {
				return true
			}

			// Check for absolute paths outside cwd
			if filepath.IsAbs(arg) {
				absPath := filepath.Clean(arg)
				if !strings.HasPrefix(absPath, cwd) {
					return true
				}
				continue
			}

			// Check for relative paths that escape cwd (e.g., ../foo, /etc/passwd)
			if strings.HasPrefix(arg, "..") {
				// Resolve the path relative to cwd
				absPath := filepath.Join(cwd, arg)
				absPath = filepath.Clean(absPath)
				if !strings.HasPrefix(absPath, cwd) {
					return true
				}
			}

			// Check for home directory expansion
			if strings.HasPrefix(arg, "~") {
				home, err := os.UserHomeDir()
				if err == nil && !strings.HasPrefix(home, cwd) {
					return true
				}
			}
		}
	}

	return false
}

// AllowlistKey generates the key for exact allowlist lookup.
func AllowlistKey(toolName string, args map[string]any) string {
	if toolName == "bash" {
		if cmd, ok := args["command"].(string); ok {
			return fmt.Sprintf("bash:%s", cmd)
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

	// For bash commands, check prefix matches with hierarchical path support
	if toolName == "bash" {
		if cmd, ok := args["command"].(string); ok {
			prefix := extractBashPrefix(cmd)
			if prefix != "" {
				// Check exact prefix match first
				if a.prefixes[prefix] {
					return true
				}
				// Check hierarchical match: if any stored prefix is a parent of current prefix
				// e.g., stored "cat:tools/" should match current "cat:tools/subdir/"
				if a.matchesHierarchicalPrefix(prefix) {
					return true
				}
			}
		}
	}

	// Check if tool itself is allowed (non-bash)
	if toolName != "bash" && a.allowlist[toolName] {
		return true
	}

	return false
}

// matchesHierarchicalPrefix checks if the given prefix matches any stored prefix hierarchically.
// For example, if "cat:tools/" is stored, it will match "cat:tools/subdir/" or "cat:tools/a/b/c/".
func (a *ApprovalManager) matchesHierarchicalPrefix(currentPrefix string) bool {
	// Split prefix into command and path parts (format: "cmd:path/")
	colonIdx := strings.Index(currentPrefix, ":")
	if colonIdx == -1 {
		return false
	}
	currentCmd := currentPrefix[:colonIdx]
	currentPath := currentPrefix[colonIdx+1:]

	for storedPrefix := range a.prefixes {
		storedColonIdx := strings.Index(storedPrefix, ":")
		if storedColonIdx == -1 {
			continue
		}
		storedCmd := storedPrefix[:storedColonIdx]
		storedPath := storedPrefix[storedColonIdx+1:]

		// Commands must match exactly
		if currentCmd != storedCmd {
			continue
		}

		// Check if current path starts with stored path (hierarchical match)
		// e.g., "tools/subdir/" starts with "tools/"
		if strings.HasPrefix(currentPath, storedPath) {
			return true
		}
	}

	return false
}

// AddToAllowlist adds a tool/command to the session allowlist.
// For bash commands, it adds the prefix pattern instead of exact command.
func (a *ApprovalManager) AddToAllowlist(toolName string, args map[string]any) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if toolName == "bash" {
		if cmd, ok := args["command"].(string); ok {
			prefix := extractBashPrefix(cmd)
			if prefix != "" {
				a.prefixes[prefix] = true
				return
			}
			// Fall back to exact match if no prefix extracted
			a.allowlist[fmt.Sprintf("bash:%s", cmd)] = true
			return
		}
	}
	a.allowlist[toolName] = true
}

// RequestApproval prompts the user for approval to execute a tool.
// Returns the decision and optional deny reason.
func (a *ApprovalManager) RequestApproval(toolName string, args map[string]any) (ApprovalResult, error) {
	// Format tool info for display
	toolDisplay := formatToolDisplay(toolName, args)

	// Enter raw mode for interactive selection
	fd := int(os.Stdin.Fd())
	oldState, err := term.MakeRaw(fd)
	if err != nil {
		// Fallback to simple input if terminal control fails
		return a.fallbackApproval(toolDisplay)
	}

	// Flush any pending stdin input before starting selector
	// This prevents buffered input from causing double-press issues
	flushStdin(fd)

	isWarning := false
	var warningMsg string
	var allowlistInfo string
	if toolName == "bash" {
		if cmd, ok := args["command"].(string); ok {
			if isCommandOutsideCwd(cmd) {
				isWarning = true
				warningMsg = "command targets paths outside project"
			}
			if prefix := extractBashPrefix(cmd); prefix != "" {
				colonIdx := strings.Index(prefix, ":")
				if colonIdx != -1 {
					cmdName := prefix[:colonIdx]
					dirPath := prefix[colonIdx+1:]
					if dirPath != "./" {
						allowlistInfo = fmt.Sprintf("%s in %s directory (includes subdirs)", cmdName, dirPath)
					} else {
						allowlistInfo = fmt.Sprintf("%s in %s directory", cmdName, dirPath)
					}
				}
			}
		}
	}

	// Run interactive selector
	selected, denyReason, err := runSelector(fd, oldState, toolDisplay, isWarning, warningMsg, allowlistInfo)
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
	var sb strings.Builder
	displayName := ToolDisplayName(toolName)

	// For bash, show command directly
	if toolName == "bash" {
		if cmd, ok := args["command"].(string); ok {
			sb.WriteString(fmt.Sprintf("Tool: %s\n", displayName))
			sb.WriteString(fmt.Sprintf("Command: %s", cmd))
			return sb.String()
		}
	}

	// For web search, show query and internet notice
	if toolName == "web_search" {
		if query, ok := args["query"].(string); ok {
			sb.WriteString(fmt.Sprintf("Tool: %s\n", displayName))
			sb.WriteString(fmt.Sprintf("Query: %s\n", query))
			sb.WriteString("Uses internet via ollama.com")
			return sb.String()
		}
	}

	// For web fetch, show URL and internet notice
	if toolName == "web_fetch" {
		if url, ok := args["url"].(string); ok {
			sb.WriteString(fmt.Sprintf("Tool: %s\n", displayName))
			sb.WriteString(fmt.Sprintf("URL: %s\n", url))
			sb.WriteString("Uses internet via ollama.com")
			return sb.String()
		}
	}

	// Generic display
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
	toolDisplay    string
	selected       int
	totalLines     int
	termWidth      int
	termHeight     int
	boxWidth       int
	innerWidth     int
	denyReason     string // deny reason (always visible in box)
	isWarning      bool   // true if command has warning
	warningMessage string // dynamic warning message to display
	allowlistInfo  string // show what will be allowlisted (for "Allow for this session" option)
}

// runSelector runs the interactive selector and returns the selected index and optional deny reason.
// If isWarning is true, the box is rendered in red to indicate the command targets paths outside cwd.
func runSelector(fd int, oldState *term.State, toolDisplay string, isWarning bool, warningMessage string, allowlistInfo string) (int, string, error) {
	state := &selectorState{
		toolDisplay:    toolDisplay,
		selected:       0,
		isWarning:      isWarning,
		warningMessage: warningMessage,
		allowlistInfo:  allowlistInfo,
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

	numOptions := len(optionLabels)

	for {
		// Read input
		buf := make([]byte, 8)
		n, err := os.Stdin.Read(buf)
		if err != nil {
			clearSelectorBox(state)
			return 2, "", err
		}

		// Process input byte by byte
		for i := 0; i < n; i++ {
			ch := buf[i]

			// Check for escape sequences (arrow keys)
			if ch == 27 && i+2 < n && buf[i+1] == '[' {
				oldSelected := state.selected
				switch buf[i+2] {
				case 'A': // Up arrow
					if state.selected > 0 {
						state.selected--
					}
				case 'B': // Down arrow
					if state.selected < numOptions-1 {
						state.selected++
					}
				}
				if oldSelected != state.selected {
					updateSelectorOptions(state)
				}
				i += 2 // Skip the rest of escape sequence
				continue
			}

			switch {
			// Ctrl+C - cancel
			case ch == 3:
				clearSelectorBox(state)
				return -1, "", nil // -1 indicates cancelled

			// Enter key - confirm selection
			case ch == 13:
				clearSelectorBox(state)
				if state.selected == 2 { // Deny
					return 2, state.denyReason, nil
				}
				return state.selected, "", nil

			// Number keys 1-3 for quick select
			case ch >= '1' && ch <= '3':
				selected := int(ch - '1')
				clearSelectorBox(state)
				if selected == 2 { // Deny
					return 2, state.denyReason, nil
				}
				return selected, "", nil

			// Backspace - delete from reason (UTF-8 safe)
			case ch == 127 || ch == 8:
				if len(state.denyReason) > 0 {
					runes := []rune(state.denyReason)
					state.denyReason = string(runes[:len(runes)-1])
					updateReasonInput(state)
				}

			// Escape - clear reason
			case ch == 27:
				if len(state.denyReason) > 0 {
					state.denyReason = ""
					updateReasonInput(state)
				}

			// Printable ASCII (except 1-3 handled above) - type into reason
			case ch >= 32 && ch < 127:
				maxLen := state.innerWidth - 2
				if maxLen < 10 {
					maxLen = 10
				}
				if len(state.denyReason) < maxLen {
					state.denyReason += string(ch)
					// Auto-select Deny option when user starts typing
					if state.selected != 2 {
						state.selected = 2
						updateSelectorOptions(state)
					} else {
						updateReasonInput(state)
					}
				}
			}
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

// getHintLines returns the hint text wrapped to terminal width
func getHintLines(state *selectorState) []string {
	hint := "up/down select, enter confirm, 1-3 quick select, ctrl+c cancel"
	if state.termWidth >= len(hint)+1 {
		return []string{hint}
	}
	// Wrap hint to multiple lines
	return wrapText(hint, state.termWidth-1)
}

// calculateTotalLines calculates how many lines the selector will use
func calculateTotalLines(state *selectorState) int {
	toolLines := strings.Split(state.toolDisplay, "\n")
	hintLines := getHintLines(state)
	// warning line (if applicable) + tool lines + blank line + options + blank line + hint lines
	warningLines := 0
	if state.isWarning {
		warningLines = 2 // warning line + blank line after
	}
	return warningLines + len(toolLines) + 1 + len(optionLabels) + 1 + len(hintLines)
}

// renderSelectorBox renders the selector (minimal, no box)
func renderSelectorBox(state *selectorState) {
	toolLines := strings.Split(state.toolDisplay, "\n")
	hintLines := getHintLines(state)

	// Draw warning line if needed
	if state.isWarning {
		if state.warningMessage != "" {
			fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m %s\033[K\r\n", state.warningMessage)
		} else {
			fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m command targets paths outside project\033[K\r\n")
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
			denyLabel := "3. Deny: "
			inputDisplay := state.denyReason
			if inputDisplay == "" {
				inputDisplay = "\033[90m(optional reason)\033[0m"
			}
			if i == state.selected {
				fmt.Fprintf(os.Stderr, "  \033[1m%s\033[0m%s\033[K\r\n", denyLabel, inputDisplay)
			} else {
				fmt.Fprintf(os.Stderr, "  \033[37m%s\033[0m%s\033[K\r\n", denyLabel, inputDisplay)
			}
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

// updateSelectorOptions updates just the options portion of the selector
func updateSelectorOptions(state *selectorState) {
	hintLines := getHintLines(state)

	// Move up to the first option line
	// Cursor is at end of last hint line, need to go up:
	// (hint lines - 1) + 1 (blank line) + numOptions
	linesToMove := len(hintLines) - 1 + 1 + len(optionLabels)
	fmt.Fprintf(os.Stderr, "\033[%dA\r", linesToMove)

	for i, label := range optionLabels {
		if i == 2 {
			denyLabel := "3. Deny: "
			inputDisplay := state.denyReason
			if inputDisplay == "" {
				inputDisplay = "\033[90m(optional reason)\033[0m"
			}
			if i == state.selected {
				fmt.Fprintf(os.Stderr, "  \033[1m%s\033[0m%s\033[K\r\n", denyLabel, inputDisplay)
			} else {
				fmt.Fprintf(os.Stderr, "  \033[37m%s\033[0m%s\033[K\r\n", denyLabel, inputDisplay)
			}
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

	// Blank line + hint
	fmt.Fprintf(os.Stderr, "\033[K\r\n")
	for i, line := range hintLines {
		if i == len(hintLines)-1 {
			fmt.Fprintf(os.Stderr, "\033[90m%s\033[0m\033[K", line)
		} else {
			fmt.Fprintf(os.Stderr, "\033[90m%s\033[0m\033[K\r\n", line)
		}
	}
}

// updateReasonInput updates just the Deny option line (which contains the reason input)
func updateReasonInput(state *selectorState) {
	hintLines := getHintLines(state)

	// Move up to the Deny line (3rd option, index 2)
	// Cursor is at end of last hint line, need to go up:
	// (hint lines - 1) + 1 (blank line) + 1 (Deny is last option)
	linesToMove := len(hintLines) - 1 + 1 + 1
	fmt.Fprintf(os.Stderr, "\033[%dA\r", linesToMove)

	// Redraw Deny line with reason
	denyLabel := "3. Deny: "
	inputDisplay := state.denyReason
	if inputDisplay == "" {
		inputDisplay = "\033[90m(optional reason)\033[0m"
	}
	if state.selected == 2 {
		fmt.Fprintf(os.Stderr, "  \033[1m%s\033[0m%s\033[K\r\n", denyLabel, inputDisplay)
	} else {
		fmt.Fprintf(os.Stderr, "  \033[37m%s\033[0m%s\033[K\r\n", denyLabel, inputDisplay)
	}

	// Blank line + hint
	fmt.Fprintf(os.Stderr, "\033[K\r\n")
	for i, line := range hintLines {
		if i == len(hintLines)-1 {
			fmt.Fprintf(os.Stderr, "\033[90m%s\033[0m\033[K", line)
		} else {
			fmt.Fprintf(os.Stderr, "\033[90m%s\033[0m\033[K\r\n", line)
		}
	}
}

// clearSelectorBox clears the selector from screen
func clearSelectorBox(state *selectorState) {
	// Clear the current line (hint line) first
	fmt.Fprint(os.Stderr, "\r\033[K")
	// Move up and clear each remaining line
	for range state.totalLines - 1 {
		fmt.Fprint(os.Stderr, "\033[A\033[K")
	}
	fmt.Fprint(os.Stderr, "\r")
}

// fallbackApproval handles approval when terminal control isn't available.
func (a *ApprovalManager) fallbackApproval(toolDisplay string) (ApprovalResult, error) {
	fmt.Fprintln(os.Stderr)
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
	displayName := ToolDisplayName(toolName)

	switch result.Decision {
	case ApprovalOnce:
		label = "Approved"
	case ApprovalAlways:
		label = "Always allowed"
	case ApprovalDeny:
		label = "Denied"
	}

	// Format based on tool type
	if toolName == "bash" {
		if cmd, ok := args["command"].(string); ok {
			// Truncate long commands
			if len(cmd) > 40 {
				cmd = cmd[:37] + "..."
			}
			return fmt.Sprintf("\033[1m%s:\033[0m %s: %s", label, displayName, cmd)
		}
	}

	if toolName == "web_search" {
		if query, ok := args["query"].(string); ok {
			// Truncate long queries
			if len(query) > 40 {
				query = query[:37] + "..."
			}
			return fmt.Sprintf("\033[1m%s:\033[0m %s: %s", label, displayName, query)
		}
	}

	if toolName == "web_fetch" {
		if url, ok := args["url"].(string); ok {
			// Truncate long URLs
			if len(url) > 50 {
				url = url[:47] + "..."
			}
			return fmt.Sprintf("\033[1m%s:\033[0m %s: %s", label, displayName, url)
		}
	}

	return fmt.Sprintf("\033[1m%s:\033[0m %s", label, displayName)
}

// FormatDenyResult returns the tool result message when a tool is denied.
func FormatDenyResult(toolName string, reason string) string {
	if reason != "" {
		return fmt.Sprintf("User denied execution of %s. Reason: %s", toolName, reason)
	}
	return fmt.Sprintf("User denied execution of %s.", toolName)
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
