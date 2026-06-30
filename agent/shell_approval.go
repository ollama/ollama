package agent

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
)

func evaluateShellApproval(req ApprovalRequest) ApprovalEvaluation {
	command, ok := stringApprovalArg(req.Args, "command")
	if !ok || strings.TrimSpace(command) == "" {
		return denyApproval(req.ToolName, ApprovalRiskHigh, "missing command argument")
	}

	risk, reasons := classifyBashCommand(command)
	if len(reasons) == 0 {
		reasons = append(reasons, "runs shell commands")
	}
	return ApprovalEvaluation{
		RequirePrompt: true,
		Risk:          risk,
		Summary:       fmt.Sprintf("%s wants to run a command", toolDisplayName(req.ToolName)),
		Reasons:       reasons,
	}
}

// Shell approval is static analysis of model-generated commands, not a sandbox.
// Globs, aliases, environment, and runtime shell state are intentionally out of scope.
func classifyBashCommand(command string) (ApprovalRisk, []string) {
	classifier := bashClassifier{risk: ApprovalRiskMedium}

	tokens, scannerReasons, scannerHigh := scanBashTokens(command)
	for _, reason := range scannerReasons {
		classifier.addReason(reason)
	}
	if scannerHigh {
		classifier.high = true
	}

	if bashFunctionDeclPattern.MatchString(command) {
		classifier.addReason("defines shell functions")
		classifier.high = true
	}
	if bashSubshellPattern.MatchString(command) {
		classifier.addReason("uses a subshell")
		classifier.high = true
	}

	for _, token := range tokens {
		if token.kind != bashTokenOperator {
			continue
		}
		switch token.value {
		case "&&", "||", "|":
			classifier.addReason("uses shell control operator " + token.value)
			classifier.high = true
		case ";", "\n":
			classifier.addReason("uses shell statement separator")
			classifier.high = true
		case "&":
			classifier.addReason("runs a command in the background")
			classifier.high = true
		case "$(", "`":
			classifier.addReason("uses command substitution")
			classifier.high = true
		case "<(", ">(":
			classifier.addReason("uses process substitution")
			classifier.high = true
		case ">", ">>", ">|", ">&", "&>":
			classifier.addReason("writes or redirects files")
			classifier.high = true
		case "<", "<<", "<&":
			classifier.addReason("uses shell redirection")
		}
	}

	for _, args := range bashCommandCalls(tokens) {
		classifier.addCall(args)
	}

	if classifier.high {
		classifier.risk = ApprovalRiskHigh
	}
	return classifier.risk, classifier.reasons
}

type bashClassifier struct {
	risk    ApprovalRisk
	high    bool
	reasons []string
}

func (c *bashClassifier) addReason(reason string) {
	if reason == "" || slices.Contains(c.reasons, reason) {
		return
	}
	c.reasons = append(c.reasons, reason)
}

func (c *bashClassifier) addCall(args []string) {
	args = shellCommandArgs(args)
	if len(args) == 0 {
		return
	}
	if isDynamicCommandName(args[0]) {
		c.addReason("uses dynamic command name")
		c.high = true
	}
	name := shellCommandBase(args[0])
	switch name {
	case "cd":
		c.addReason("changes directory")
		c.high = true
	case "eval":
		c.addReason("evaluates shell code")
		c.high = true
	case "source", ".":
		c.addReason("sources shell code")
		c.high = true
	case "exec":
		c.addReason("replaces the shell process")
		c.high = true
	case "sudo":
		c.addReason("runs with elevated privileges")
		c.high = true
	case "rm":
		if hasAnyFlag(args[1:], "r", "R", "recursive") || hasAnyFlag(args[1:], "f", "force") {
			c.addReason("removes files destructively")
			c.high = true
		}
	case "git":
		if isGitResetHard(args) {
			c.addReason("runs destructive git reset")
			c.high = true
		}
		if isGitCleanDestructive(args) {
			c.addReason("runs destructive git clean")
			c.high = true
		}
	case "find":
		c.addFindReasons(args)
	case "chmod", "chown":
		if hasAnyFlag(args[1:], "R", "recursive") {
			c.addReason("changes permissions or ownership recursively")
			c.high = true
		}
	case "dd", "diskutil":
		c.addReason("can write directly to disks")
		c.high = true
	case "mkfs", "newfs":
		c.addReason("formats filesystems")
		c.high = true
	case "curl", "wget":
		c.addReason("downloads remote content")
	case "sh", "bash", "zsh":
		if len(args) > 1 {
			c.addReason("runs a shell interpreter")
			c.high = true
		}
	}
}

func (c *bashClassifier) addFindReasons(args []string) {
	for i := 1; i < len(args); i++ {
		switch args[i] {
		case "-delete":
			c.addReason("deletes files via find")
			c.high = true
		case "-exec", "-execdir":
			c.addReason("executes commands via find")
			c.high = true
			end := i + 1
			for end < len(args) && args[end] != ";" && args[end] != `\;` && args[end] != "+" {
				end++
			}
			if end > i+1 {
				c.addCall(args[i+1 : end])
			}
			i = end
		}
	}
}

func hasAnyFlag(args []string, flags ...string) bool {
	for _, arg := range args {
		if arg == "--" {
			return false
		}
		if !strings.HasPrefix(arg, "-") {
			continue
		}
		for _, flag := range flags {
			short := len(flag) == 1
			if short && strings.HasPrefix(arg, "-") && !strings.HasPrefix(arg, "--") && strings.Contains(arg[1:], flag) {
				return true
			}
			if arg == "--"+flag {
				return true
			}
		}
	}
	return false
}

func isGitResetHard(args []string) bool {
	subcommand := gitSubcommandArgs(args)
	return len(subcommand) >= 2 && subcommand[0] == "reset" && slices.Contains(subcommand[1:], "--hard")
}

func isGitCleanDestructive(args []string) bool {
	subcommand := gitSubcommandArgs(args)
	if len(subcommand) < 1 || subcommand[0] != "clean" {
		return false
	}
	return hasAnyFlag(subcommand[1:], "f", "force") && (hasAnyFlag(subcommand[1:], "d") || hasAnyFlag(subcommand[1:], "x", "X"))
}

func gitSubcommandArgs(args []string) []string {
	for i := 1; i < len(args); i++ {
		arg := args[i]
		if arg == "--" {
			return args[i+1:]
		}
		switch {
		case arg == "-C", arg == "-c", arg == "--git-dir", arg == "--work-tree":
			i++
			continue
		case strings.HasPrefix(arg, "-C"), strings.HasPrefix(arg, "-c"):
			continue
		case strings.HasPrefix(arg, "--git-dir="), strings.HasPrefix(arg, "--work-tree="):
			continue
		case strings.HasPrefix(arg, "-"):
			continue
		default:
			return args[i:]
		}
	}
	return nil
}

type bashTokenKind int

const (
	bashTokenWord bashTokenKind = iota
	bashTokenOperator
)

type bashToken struct {
	kind  bashTokenKind
	value string
}

var (
	bashFunctionDeclPattern = regexp.MustCompile(`(?m)(^|[;&|[:space:]])(?:function[[:space:]]+)?[A-Za-z_][A-Za-z0-9_]*[[:space:]]*(?:\(\)[[:space:]]*)?\{`)
	bashSubshellPattern     = regexp.MustCompile(`(?m)(^|[;&|[:space:]])\(`)
)

func scanBashTokens(command string) ([]bashToken, []string, bool) {
	var tokens []bashToken
	var reasons []string
	var word strings.Builder
	var quote byte
	escaped := false
	high := false

	flushWord := func() {
		if word.Len() == 0 {
			return
		}
		tokens = append(tokens, bashToken{kind: bashTokenWord, value: word.String()})
		word.Reset()
	}
	addOperator := func(op string) {
		tokens = append(tokens, bashToken{kind: bashTokenOperator, value: op})
	}

	for i := 0; i < len(command); i++ {
		ch := command[i]
		if escaped {
			word.WriteByte(ch)
			escaped = false
			continue
		}
		if ch == '\\' && quote != '\'' {
			escaped = true
			continue
		}

		if quote == '\'' {
			if ch == '\'' {
				quote = 0
			} else {
				word.WriteByte(ch)
			}
			continue
		}
		if quote == '"' {
			switch {
			case ch == '"':
				quote = 0
			case ch == '`':
				addOperator("`")
				word.WriteByte(ch)
			case ch == '$' && i+1 < len(command) && command[i+1] == '(':
				addOperator("$(")
				word.WriteString("$(")
				i++
			default:
				word.WriteByte(ch)
			}
			continue
		}

		if ch == '\'' || ch == '"' {
			quote = ch
			continue
		}
		if ch == '`' {
			addOperator("`")
			word.WriteByte(ch)
			continue
		}
		if ch == '$' && i+1 < len(command) && command[i+1] == '(' {
			addOperator("$(")
			word.WriteString("$(")
			i++
			continue
		}
		if (ch == '<' || ch == '>') && i+1 < len(command) && command[i+1] == '(' {
			flushWord()
			addOperator(string([]byte{ch, '('}))
			i++
			continue
		}
		if ch == '\n' {
			flushWord()
			addOperator("\n")
			continue
		}
		if ch == ' ' || ch == '\t' || ch == '\r' {
			flushWord()
			continue
		}

		switch ch {
		case '&':
			flushWord()
			switch {
			case i+1 < len(command) && command[i+1] == '&':
				addOperator("&&")
				i++
			case i+1 < len(command) && command[i+1] == '>':
				addOperator("&>")
				i++
			default:
				addOperator("&")
			}
		case '|':
			flushWord()
			if i+1 < len(command) && command[i+1] == '|' {
				addOperator("||")
				i++
			} else {
				addOperator("|")
			}
		case ';':
			flushWord()
			addOperator(";")
		case '<':
			flushWord()
			if i+1 < len(command) && command[i+1] == '<' {
				addOperator("<<")
				i++
			} else if i+1 < len(command) && command[i+1] == '&' {
				addOperator("<&")
				i++
			} else {
				addOperator("<")
			}
		case '>':
			flushWord()
			if i+1 < len(command) && command[i+1] == '>' {
				addOperator(">>")
				i++
			} else if i+1 < len(command) && command[i+1] == '|' {
				addOperator(">|")
				i++
			} else if i+1 < len(command) && command[i+1] == '&' {
				addOperator(">&")
				i++
			} else {
				addOperator(">")
			}
		default:
			word.WriteByte(ch)
		}
	}
	if escaped || quote != 0 {
		reasons = append(reasons, "could not parse shell command")
		high = true
	}
	flushWord()
	return tokens, reasons, high
}

func bashCommandCalls(tokens []bashToken) [][]string {
	var calls [][]string
	var current []string
	flush := func() {
		if len(current) == 0 {
			return
		}
		calls = append(calls, current)
		current = nil
	}
	for _, token := range tokens {
		if token.kind == bashTokenWord {
			current = append(current, token.value)
			continue
		}
		if isBashCommandBoundary(token.value) {
			flush()
		}
	}
	flush()
	return calls
}

func isBashCommandBoundary(op string) bool {
	switch op {
	case "&&", "||", "|", ";", "&", "\n":
		return true
	default:
		return false
	}
}

func shellCommandArgs(args []string) []string {
	for len(args) > 0 && isShellAssignment(args[0]) {
		args = args[1:]
	}
	return args
}

func isShellAssignment(word string) bool {
	name, _, ok := strings.Cut(word, "=")
	if !ok || name == "" {
		return false
	}
	for i := range len(name) {
		ch := name[i]
		if i == 0 {
			if !((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch == '_') {
				return false
			}
			continue
		}
		if !((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_') {
			return false
		}
	}
	return true
}

func isDynamicCommandName(name string) bool {
	return strings.HasPrefix(name, "$") || strings.HasPrefix(name, "`") || strings.Contains(name, "$(") || strings.Contains(name, "`")
}

func shellCommandBase(name string) string {
	name = strings.TrimSpace(name)
	if i := strings.LastIndexAny(name, `/\`); i >= 0 && i+1 < len(name) {
		name = name[i+1:]
	}
	return name
}
