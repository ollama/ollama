package agent

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	tree_sitter "github.com/tree-sitter/go-tree-sitter"
	tree_sitter_bash "github.com/tree-sitter/tree-sitter-bash/bindings/go"
)

// AnalysisResult contains the result of analyzing a bash command.
type AnalysisResult struct {
	Command    string
	Suspicious bool
	Reason     string
	Reasons    []string
}

// ParseResult contains the parsed AST for a bash command.
type ParseResult struct {
	Root   *tree_sitter.Node
	Source []byte
	Tree   *tree_sitter.Tree
}

// PatternType defines how a suspicious pattern should be matched.
type PatternType int

const (
	PatternTypeCommand PatternType = iota
	PatternTypeArgs
	PatternTypeSubstring
	PatternTypeAST
)

// SuspiciousPattern defines a single suspicious pattern.
type SuspiciousPattern struct {
	Command     string
	PatternType PatternType
	Args        []string
	Substring   string
	Reason      string
}

var whitespacePattern = regexp.MustCompile(`\s+`)

var suspiciousPatterns = []SuspiciousPattern{
	{
		Command:     "mkfs",
		PatternType: PatternTypeCommand,
		Reason:      "Creates a filesystem",
	},
	{
		Command:     "shred",
		PatternType: PatternTypeCommand,
		Reason:      "Securely deletes data",
	},
	{
		Command:     "dd",
		PatternType: PatternTypeCommand,
		Reason:      "Reads or overwrites disks",
	},
	{
		Command:     "mkfifo",
		PatternType: PatternTypeCommand,
		Reason:      "Creates a named pipe",
	},
	{
		Command:     "history",
		PatternType: PatternTypeCommand,
		Reason:      "Reads shell history",
	},
	{
		Command:     "nc",
		PatternType: PatternTypeCommand,
		Reason:      "Opens raw network connections",
	},
	{
		Command:     "netcat",
		PatternType: PatternTypeCommand,
		Reason:      "Opens raw network connections",
	},
	{
		Command:     "scp",
		PatternType: PatternTypeCommand,
		Reason:      "Copies files over SSH",
	},
	{
		Command:     "rsync",
		PatternType: PatternTypeCommand,
		Reason:      "Syncs files remotely",
	},
	{
		Command:     "bash",
		PatternType: PatternTypeCommand,
		Reason:      "Spawns a shell",
	},
	{
		Command:     "sh",
		PatternType: PatternTypeCommand,
		Reason:      "Spawns a shell",
	},
	{
		Command:     "zsh",
		PatternType: PatternTypeCommand,
		Reason:      "Spawns a shell",
	},
	{
		Command:     "eval",
		PatternType: PatternTypeCommand,
		Reason:      "Executes shell code",
	},
	{
		Command:     "exec",
		PatternType: PatternTypeCommand,
		Reason:      "Replaces process with command",
	},
	{
		Command:     "xargs",
		PatternType: PatternTypeCommand,
		Reason:      "Builds and runs commands",
	},
	{
		Command:     "alias",
		PatternType: PatternTypeCommand,
		Reason:      "Redefines shell commands",
	},
	{
		Command:     "arp",
		PatternType: PatternTypeCommand,
		Reason:      "Probes the local network",
	},
	{
		Command:     "users",
		PatternType: PatternTypeCommand,
		Reason:      "Reveals user identity",
	},
	{
		Command:     "netstat",
		PatternType: PatternTypeCommand,
		Reason:      "Lists network connections",
	},
	{
		Command:     "uname",
		PatternType: PatternTypeCommand,
		Reason:      "Reveals OS details",
	},
	{
		Command:     "groups",
		PatternType: PatternTypeCommand,
		Reason:      "Lists users and groups",
	},
	{
		Command:     "lsmod",
		PatternType: PatternTypeCommand,
		Reason:      "Lists loaded kernel modules",
	},
	{
		Command:     "whoami",
		PatternType: PatternTypeCommand,
		Reason:      "Reveals user identity",
	},
	{
		Command:     "id",
		PatternType: PatternTypeCommand,
		Reason:      "Reveals user and group IDs",
	},
	{
		Command:     "nmap",
		PatternType: PatternTypeCommand,
		Reason:      "Scans network hosts",
	},
	{
		Command:     "tftp",
		PatternType: PatternTypeCommand,
		Reason:      "Transfers files over TFTP",
	},
	{
		Command:     "insmod",
		PatternType: PatternTypeCommand,
		Reason:      "Loads a kernel module",
	},
	{
		Command:     "modprobe",
		PatternType: PatternTypeCommand,
		Reason:      "Loads a kernel module",
	},
	{
		Command:     "useradd",
		PatternType: PatternTypeCommand,
		Reason:      "Creates a user account",
	},
	{
		Command:     "usermod",
		PatternType: PatternTypeCommand,
		Reason:      "Modifies a user account",
	},
	{
		Command:     "crontab",
		PatternType: PatternTypeCommand,
		Reason:      "Schedules a cron job",
	},
	{
		Command:     "tcpdump",
		PatternType: PatternTypeCommand,
		Reason:      "Captures network traffic",
	},
	{
		Command:     "kill",
		PatternType: PatternTypeCommand,
		Reason:      "Terminates processes",
	},
	{
		Command:     "pkill",
		PatternType: PatternTypeCommand,
		Reason:      "Terminates processes",
	},
	{
		Command:     "sudo",
		PatternType: PatternTypeCommand,
		Reason:      "Escalates privileges",
	},
	{
		Command:     "su",
		PatternType: PatternTypeCommand,
		Reason:      "Escalates privileges",
	},
	{
		Command:     "doas",
		PatternType: PatternTypeCommand,
		Reason:      "Escalates privileges",
	},
	{
		Command:     "rm",
		PatternType: PatternTypeArgs,
		Args:        []string{"-r", "-f"},
		Reason:      "Force-deletes recursively",
	},
	{
		Command:     "rm",
		PatternType: PatternTypeArgs,
		Args:        []string{"--recursive", "--force"},
		Reason:      "Force-deletes recursively",
	},
	{
		Command:     "rm",
		PatternType: PatternTypeArgs,
		Args:        []string{"-r", "--force"},
		Reason:      "Force-deletes recursively",
	},
	{
		Command:     "rm",
		PatternType: PatternTypeArgs,
		Args:        []string{"--recursive", "-f"},
		Reason:      "Force-deletes recursively",
	},
	{
		Command:     "chmod",
		PatternType: PatternTypeArgs,
		Args:        []string{"777"},
		Reason:      "Makes files world-writable",
	},
	{
		Command:     "chmod",
		PatternType: PatternTypeArgs,
		Args:        []string{"0777"},
		Reason:      "Makes files world-writable",
	},
	{
		Command:     "chmod",
		PatternType: PatternTypeArgs,
		Args:        []string{"a+rwx"},
		Reason:      "Makes files world-writable",
	},
	{
		Command:     "chmod",
		PatternType: PatternTypeArgs,
		Args:        []string{"+s"},
		Reason:      "Sets SUID/SGID bits",
	},
	{
		Command:     "chown",
		PatternType: PatternTypeArgs,
		Args:        []string{":root"},
		Reason:      "Changes owner to root",
	},
	{
		Command:     "chown",
		PatternType: PatternTypeArgs,
		Args:        []string{":0"},
		Reason:      "Changes owner to root",
	},
	{
		Command:     "chgrp",
		PatternType: PatternTypeArgs,
		Args:        []string{"root"},
		Reason:      "Changes group to root",
	},
	{
		Command:     "chgrp",
		PatternType: PatternTypeArgs,
		Args:        []string{"0"},
		Reason:      "Changes group to root",
	},
	{
		Command:     "curl",
		PatternType: PatternTypeArgs,
		Args:        []string{"-d"},
		Reason:      "Sends request data",
	},
	{
		Command:     "curl",
		PatternType: PatternTypeArgs,
		Args:        []string{"--data"},
		Reason:      "Sends request data",
	},
	{
		Command:     "curl",
		PatternType: PatternTypeArgs,
		Args:        []string{"-X", "POST"},
		Reason:      "Sends a POST request",
	},
	{
		Command:     "curl",
		PatternType: PatternTypeArgs,
		Args:        []string{"-X", "PUT"},
		Reason:      "Sends a PUT request",
	},
	{
		Command:     "wget",
		PatternType: PatternTypeArgs,
		Args:        []string{"--post"},
		Reason:      "Sends a POST request",
	},
	{
		Command:     "wget",
		PatternType: PatternTypeArgs,
		Args:        []string{"--post-data"},
		Reason:      "Sends a POST request",
	},
	{
		Command:     "systemctl",
		PatternType: PatternTypeArgs,
		Args:        []string{"stop"},
		Reason:      "Stops a service",
	},
	{
		Command:     "systemctl",
		PatternType: PatternTypeArgs,
		Args:        []string{"disable"},
		Reason:      "Disables a service",
	},
	{
		Command:     "find",
		PatternType: PatternTypeArgs,
		Args:        []string{"-name", "*password*"},
		Reason:      "Searches for password files",
	},
	{
		Command:     "find",
		PatternType: PatternTypeArgs,
		Args:        []string{"-name", "*secret*"},
		Reason:      "Searches for secret files",
	},
	{
		Command:     "find",
		PatternType: PatternTypeArgs,
		Args:        []string{"-name", "*credential*"},
		Reason:      "Searches for credential files",
	},
	{
		Command:     "find",
		PatternType: PatternTypeArgs,
		Args:        []string{"-name", "*token*"},
		Reason:      "Searches for token files",
	},
	{
		Command:     "find",
		PatternType: PatternTypeArgs,
		Args:        []string{"-name", "*.pem"},
		Reason:      "Searches for PEM files",
	},
	{
		Command:     "find",
		PatternType: PatternTypeArgs,
		Args:        []string{"-name", "*.key"},
		Reason:      "Searches for key files",
	},
	{
		Command:     "find",
		PatternType: PatternTypeArgs,
		Args:        []string{"-name", "*.p12"},
		Reason:      "Searches for P12 files",
	},
	{
		Command:     "find",
		PatternType: PatternTypeArgs,
		Args:        []string{"-name", "*id_rsa*"},
		Reason:      "Searches for RSA keys",
	},
	{
		Command:     "find",
		PatternType: PatternTypeArgs,
		Args:        []string{"-name", "*id_dsa*"},
		Reason:      "Searches for DSA keys",
	},
	{
		Command:     "find",
		PatternType: PatternTypeArgs,
		Args:        []string{"-name", "*id_ecdsa*"},
		Reason:      "Searches for ECDSA keys",
	},
	{
		Command:     "find",
		PatternType: PatternTypeArgs,
		Args:        []string{"-name", "*id_ed25519*"},
		Reason:      "Searches for Ed25519 keys",
	},
	{
		Command:     "grep",
		PatternType: PatternTypeArgs,
		Args:        []string{"-i", "password"},
		Reason:      "Searches for passwords",
	},
	{
		Command:     "grep",
		PatternType: PatternTypeArgs,
		Args:        []string{"-i", "secret"},
		Reason:      "Searches for secrets",
	},
	{
		Command:     "grep",
		PatternType: PatternTypeArgs,
		Args:        []string{"-i", "credential"},
		Reason:      "Searches for credentials",
	},
	{
		Command:     "grep",
		PatternType: PatternTypeArgs,
		Args:        []string{"-i", "token"},
		Reason:      "Searches for tokens",
	},
	{
		Command:     "grep",
		PatternType: PatternTypeArgs,
		Args:        []string{"-i", "api_key"},
		Reason:      "Searches for API keys",
	},
	{
		Command:     "grep",
		PatternType: PatternTypeArgs,
		Args:        []string{"-i", "apikey"},
		Reason:      "Searches for API keys",
	},
	{
		Command:     "grep",
		PatternType: PatternTypeArgs,
		Args:        []string{"-i", "private_key"},
		Reason:      "Searches for private keys",
	},
	{
		Command:     "grep",
		PatternType: PatternTypeArgs,
		Args:        []string{"-i", "access_key"},
		Reason:      "Searches for access keys",
	},
	{
		Command:     "aws",
		PatternType: PatternTypeArgs,
		Args:        []string{"sts", "get-caller-identity"},
		Reason:      "Reads AWS identity",
	},
	{
		Command:     "aws",
		PatternType: PatternTypeArgs,
		Args:        []string{"iam", "add-user-to-group"},
		Reason:      "Modifies IAM users",
	},
	{
		Command:     "aws",
		PatternType: PatternTypeArgs,
		Args:        []string{"iam", "attach-user-policy"},
		Reason:      "Attaches IAM policies",
	},
	{
		Command:     "aws",
		PatternType: PatternTypeArgs,
		Args:        []string{"iam", "put-user-policy"},
		Reason:      "Creates IAM policies",
	},
	{
		Command:     "aws",
		PatternType: PatternTypeArgs,
		Args:        []string{"iam", "create-access-key"},
		Reason:      "Creates IAM access keys",
	},
	{
		Command:     "aws",
		PatternType: PatternTypeArgs,
		Args:        []string{"iam", "delete-access-key"},
		Reason:      "Deletes IAM access keys",
	},
	{
		Command:     "aws",
		PatternType: PatternTypeArgs,
		Args:        []string{"iam", "list-users"},
		Reason:      "Lists IAM users",
	},
	{
		Command:     "aws",
		PatternType: PatternTypeArgs,
		Args:        []string{"iam", "list-roles"},
		Reason:      "Lists IAM roles",
	},
	{
		Command:     "aws",
		PatternType: PatternTypeArgs,
		Args:        []string{"iam", "get-user"},
		Reason:      "Reads IAM user details",
	},
	{
		Command:     "aws",
		PatternType: PatternTypeArgs,
		Args:        []string{"ec2", "describe-instances"},
		Reason:      "Lists EC2 instances",
	},
	{
		Command:     "aws",
		PatternType: PatternTypeArgs,
		Args:        []string{"ec2", "describe-key-pairs"},
		Reason:      "Lists EC2 key pairs",
	},
	{
		Command:     "aws",
		PatternType: PatternTypeArgs,
		Args:        []string{"ec2", "describe-security-groups"},
		Reason:      "Lists EC2 security groups",
	},
	{
		Command:     "aws",
		PatternType: PatternTypeArgs,
		Args:        []string{"s3", "ls"},
		Reason:      "Lists S3 contents",
	},
	{
		Command:     "aws",
		PatternType: PatternTypeArgs,
		Args:        []string{"s3", "cp"},
		Reason:      "Copies files with S3",
	},
	{
		Command:     "aws",
		PatternType: PatternTypeArgs,
		Args:        []string{"s3", "sync"},
		Reason:      "Syncs files with S3",
	},
	{
		Command:     "setfacl",
		PatternType: PatternTypeArgs,
		Args:        []string{"-R"},
		Reason:      "Changes ACLs recursively",
	},
	{
		Command:     "setfacl",
		PatternType: PatternTypeArgs,
		Args:        []string{"-m"},
		Reason:      "Changes ACLs",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ":(){ :|:& };:",
		Reason:      "Exhausts system processes",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ":(){ :|:& }; :",
		Reason:      "Exhausts system processes",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "/dev/sda",
		Reason:      "Accesses raw disks",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "/dev/nvme",
		Reason:      "Accesses raw disks",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "/dev/hd",
		Reason:      "Accesses raw disks",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "> /dev/",
		Reason:      "Writes to a device",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ">/dev/",
		Reason:      "Writes to a device",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "/etc/shadow",
		Reason:      "Accesses password hashes",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "/etc/passwd",
		Reason:      "Accesses account file",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "/etc/sudoers",
		Reason:      "Accesses sudo rules",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ".ssh/id_rsa",
		Reason:      "Accesses SSH private key",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ".ssh/id_dsa",
		Reason:      "Accesses SSH private key",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ".ssh/id_ecdsa",
		Reason:      "Accesses SSH private key",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ".ssh/id_ed25519",
		Reason:      "Accesses SSH private key",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ".ssh/config",
		Reason:      "Accesses SSH config",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ".aws/credentials",
		Reason:      "Accesses AWS credentials",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ".aws/config",
		Reason:      "Accesses AWS config",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ".gnupg/",
		Reason:      "Accesses GPG keys",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ".env",
		Reason:      "Accesses environment secrets",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "credentials.json",
		Reason:      "Accesses credentials file",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "secrets.json",
		Reason:      "Accesses secrets file",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "secrets.yaml",
		Reason:      "Accesses secrets file",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "secrets.yml",
		Reason:      "Accesses secrets file",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "-exec rm",
		Reason:      "Runs rm via find",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "-execdir rm",
		Reason:      "Runs rm via find",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "LD_PRELOAD=",
		Reason:      "Injects a shared library",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "/etc/ld.so.preload",
		Reason:      "Modifies preload list",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ".bashrc",
		Reason:      "Modifies shell startup",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ".bash_profile",
		Reason:      "Modifies shell startup",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ".bash_history",
		Reason:      "Accesses shell history",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "~/.bash_history",
		Reason:      "Accesses shell history",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ".zsh_history",
		Reason:      "Accesses shell history",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "~/.zsh_history",
		Reason:      "Accesses shell history",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   ".history",
		Reason:      "Accesses shell history",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "~/.history",
		Reason:      "Accesses shell history",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "chattr -i",
		Reason:      "Removes immutable flag",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "chattr +i",
		Reason:      "Adds immutable flag",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "HISTFILESIZE=0",
		Reason:      "Disables shell history",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "HISTSIZE=0",
		Reason:      "Disables shell history",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "history -c",
		Reason:      "Clears shell history",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "unset HISTFILE",
		Reason:      "Disables history file",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "service auditd stop",
		Reason:      "Stops audit logging",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "service rsyslog stop",
		Reason:      "Stops system logging",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "cat /dev/null >",
		Reason:      "Clears file contents",
	},
	{
		Command:     "*",
		PatternType: PatternTypeAST,
		Substring:   "history_redirect",
		Reason:      "Redirects to history file",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "http_proxy=",
		Reason:      "Changes HTTP proxy",
	},
	{
		Command:     "*",
		PatternType: PatternTypeSubstring,
		Substring:   "https_proxy=",
		Reason:      "Changes HTTPS proxy",
	},
}

func bashToolDisplayName(toolName string) (string, bool) {
	if toolName == "bash" {
		return "Bash", true
	}
	return "", false
}

func formatBashToolDisplay(args map[string]any) (string, bool) {
	if cmd, ok := args["command"].(string); ok {
		return fmt.Sprintf("Tool: %s\nCommand: %s", ToolDisplayName("bash"), cmd), true
	}

	return "", false
}

func formatBashApprovalResult(label string, args map[string]any) (string, bool) {
	if cmd, ok := args["command"].(string); ok {
		return fmt.Sprintf("\033[1m%s:\033[0m %s: %s", label, ToolDisplayName("bash"), truncateDisplayText(cmd, 40)), true
	}

	return "", false
}

func bashAllowlistKey(args map[string]any) (string, bool) {
	if cmd, ok := args["command"].(string); ok {
		return fmt.Sprintf("bash:%s", cmd), true
	}
	return "", false
}

func (a *ApprovalManager) isAllowedBash(args map[string]any) bool {
	if cmd, ok := args["command"].(string); ok {
		prefix := extractBashPrefix(cmd)
		if prefix != "" {
			if a.prefixes[prefix] {
				return true
			}
			if a.matchesHierarchicalPrefix(prefix) {
				return true
			}
		}
	}

	return false
}

func (a *ApprovalManager) addToAllowlistBash(args map[string]any) bool {
	cmd, ok := args["command"].(string)
	if !ok {
		return false
	}

	prefix := extractBashPrefix(cmd)
	if prefix != "" {
		a.prefixes[prefix] = true
		return true
	}

	a.allowlist[fmt.Sprintf("bash:%s", cmd)] = true
	return true
}

// BuildBashApprovalOptions gathers warnings and allowlist context for a bash prompt.
func BuildBashApprovalOptions(command string) ApprovalPromptOptions {
	opts := ApprovalPromptOptions{
		AllowlistInfo: describeBashAllowlist(command),
	}

	if analysis, err := AnalyzeCommand(command); err == nil && analysis != nil && analysis.Suspicious {
		for _, reason := range analysis.reasons() {
			opts.Warnings = append(opts.Warnings, fmt.Sprintf("command flagged as suspicious: %s", reason))
		}
	}

	if isCommandOutsideCwd(command) {
		opts.Warnings = append(opts.Warnings, "command targets paths outside project")
	}

	return opts
}

// IsSuspicious reports whether a bash command is suspicious according to the analyzer.
func IsSuspicious(command string) (bool, []string) {
	result, err := AnalyzeCommand(command)
	if result == nil {
		return false, nil
	}

	if err != nil {
		return result.Suspicious, append([]string(nil), result.reasons()...)
	}

	return result.Suspicious, append([]string(nil), result.reasons()...)
}

// ParseBash parses a bash command string and returns the AST.
func ParseBash(source string) (*ParseResult, error) {
	parser := tree_sitter.NewParser()
	defer parser.Close()

	language := tree_sitter.NewLanguage(tree_sitter_bash.Language())
	if err := parser.SetLanguage(language); err != nil {
		return nil, err
	}

	tree := parser.Parse([]byte(source), nil)
	if tree == nil {
		return nil, nil
	}

	return &ParseResult{
		Root:   tree.RootNode(),
		Source: []byte(source),
		Tree:   tree,
	}, nil
}

// AnalyzeCommand analyzes a bash command for suspicious patterns.
func AnalyzeCommand(command string) (*AnalysisResult, error) {
	result := &AnalysisResult{Command: command}

	for _, reason := range collectRedirectionReasons(command) {
		addSuspiciousReason(result, reason)
	}

	normalizedCommand := normalizeWhitespace(command)
	for _, pattern := range suspiciousPatterns {
		if pattern.PatternType != PatternTypeSubstring {
			continue
		}
		if strings.Contains(normalizedCommand, normalizeWhitespace(pattern.Substring)) {
			addSuspiciousReason(result, pattern.Reason)
		}
	}

	parseResult, err := ParseBash(command)
	if err != nil || parseResult == nil {
		return result, err
	}
	defer parseResult.Tree.Close()

	analyzeNode(parseResult.Root, parseResult.Source, result)

	return result, nil
}

func normalizeWhitespace(s string) string {
	return whitespacePattern.ReplaceAllString(s, " ")
}

func (r *AnalysisResult) reasons() []string {
	if len(r.Reasons) > 0 {
		return r.Reasons
	}
	if r.Reason == "" {
		return nil
	}
	return []string{r.Reason}
}

func addSuspiciousReason(result *AnalysisResult, reason string) {
	if reason == "" {
		return
	}

	for _, existing := range result.Reasons {
		if existing == reason {
			return
		}
	}

	result.Suspicious = true
	if result.Reason == "" {
		result.Reason = reason
	}
	result.Reasons = append(result.Reasons, reason)
}

func collectRedirectionReasons(command string) []string {
	normalized := normalizeWhitespace(command)
	re := regexp.MustCompile(`(?:^|[\s;|&])((?:\d*>>?)|(?:\d*<))\s*([^\s;|&]+)`)
	matches := re.FindAllStringSubmatch(normalized, -1)
	var reasons []string

	for _, match := range matches {
		if len(match) < 3 {
			continue
		}
		operator := match[1]
		target := normalizeRedirectTarget(match[2])

		if strings.Contains(operator, ">") && isHistoryPath(target) {
			reasons = appendReason(reasons, "Redirects to history file")
		}
		if strings.HasPrefix(target, "/dev/") {
			if strings.Contains(operator, "<") {
				reasons = appendReason(reasons, "Reads from a device")
			} else {
				reasons = appendReason(reasons, "Writes to a device")
			}
		}
	}

	return reasons
}

func normalizeRedirectTarget(target string) string {
	return strings.Trim(target, `"'`)
}

func isHistoryPath(target string) bool {
	for _, historyFile := range []string{".bash_history", ".zsh_history", ".history"} {
		if target == historyFile || target == "~/"+historyFile || strings.HasSuffix(target, "/"+historyFile) {
			return true
		}
	}

	return false
}

func appendReason(reasons []string, reason string) []string {
	for _, existing := range reasons {
		if existing == reason {
			return reasons
		}
	}
	return append(reasons, reason)
}

func analyzeNode(node *tree_sitter.Node, source []byte, result *AnalysisResult) {
	if node.Kind() == "command" {
		for _, reason := range checkCommand(node, source) {
			addSuspiciousReason(result, reason)
		}
	}

	if node.Kind() == "function_definition" {
		if reason := checkForkBomb(node, source); reason != "" {
			addSuspiciousReason(result, reason)
		}
	}

	for i := range node.ChildCount() {
		analyzeNode(node.Child(i), source, result)
	}
}

func checkForkBomb(node *tree_sitter.Node, source []byte) string {
	if node.Kind() != "function_definition" {
		return ""
	}

	functionNameNode := node.Child(0)
	if functionNameNode == nil || functionNameNode.Kind() != "word" {
		return ""
	}

	funcName := string(source[functionNameNode.StartByte():functionNameNode.EndByte()])
	if funcName != ":" {
		return ""
	}

	body := node.Child(1)
	if body == nil {
		return ""
	}

	bodyText := string(source[body.StartByte():body.EndByte()])
	if strings.Contains(bodyText, "|") && strings.Contains(bodyText, "&") {
		return "Exhausts system processes"
	}

	return ""
}

type commandInfo struct {
	Name string
	Args []string
}

func getCommandInfo(node *tree_sitter.Node, source []byte) *commandInfo {
	if node.Kind() != "command" {
		return nil
	}

	commandNameNode := node.Child(0)
	if commandNameNode == nil || commandNameNode.Kind() != "command_name" || commandNameNode.ChildCount() == 0 {
		return nil
	}

	wordNode := commandNameNode.Child(0)
	if wordNode == nil || wordNode.Kind() != "word" {
		return nil
	}

	cmdName := string(source[wordNode.StartByte():wordNode.EndByte()])
	var args []string

	for i := uint(1); i < node.ChildCount(); i++ {
		child := node.Child(i)
		switch child.Kind() {
		case "word", "number":
			args = append(args, string(source[child.StartByte():child.EndByte()]))
		case "string":
			args = append(args, extractStringContent(child, source))
		case "raw_string":
			args = append(args, extractRawStringContent(child, source))
		}
	}

	return &commandInfo{Name: cmdName, Args: args}
}

func extractStringContent(node *tree_sitter.Node, source []byte) string {
	for i := range node.ChildCount() {
		child := node.Child(i)
		if child.Kind() == "string_content" {
			return string(source[child.StartByte():child.EndByte()])
		}
	}

	return string(source[node.StartByte():node.EndByte()])
}

func extractRawStringContent(node *tree_sitter.Node, source []byte) string {
	content := string(source[node.StartByte():node.EndByte()])
	if len(content) >= 2 && content[0] == '\'' && content[len(content)-1] == '\'' {
		return content[1 : len(content)-1]
	}
	return content
}

func checkCommand(node *tree_sitter.Node, source []byte) []string {
	cmdInfo := getCommandInfo(node, source)
	if cmdInfo == nil {
		return nil
	}

	cmdName := cmdInfo.Name
	var reasons []string
	for _, pattern := range suspiciousPatterns {
		if pattern.PatternType != PatternTypeCommand || pattern.Command != cmdName {
			continue
		}
		if cmdName == "bash" || cmdName == "sh" || cmdName == "zsh" {
			for i, arg := range cmdInfo.Args {
				if arg == "-c" && i+1 < len(cmdInfo.Args) {
					return suspiciousReasonsForInnerCommand(strings.Join(cmdInfo.Args[i+1:], " "))
				}
			}
		}
		reasons = appendReason(reasons, pattern.Reason)
	}

	for _, pattern := range suspiciousPatterns {
		if pattern.PatternType != PatternTypeArgs || pattern.Command != cmdName {
			continue
		}
		if hasAllArgs(cmdInfo.Args, pattern.Args) {
			reasons = appendReason(reasons, pattern.Reason)
		}
	}

	return reasons
}

func suspiciousReasonsForInnerCommand(innerCmd string) []string {
	result, err := AnalyzeCommand(innerCmd)
	if err != nil || result == nil {
		return nil
	}
	return result.reasons()
}

func hasAllArgs(args []string, values []string) bool {
	for _, value := range values {
		if !hasArg(args, value) {
			return false
		}
	}
	return true
}

func hasArg(args []string, value string) bool {
	for _, arg := range args {
		if arg == value {
			return true
		}
		if len(arg) >= 2 && arg[0] == '-' && len(value) >= 2 && value[0] == '-' && strings.Contains(arg, value[1:]) {
			return true
		}
	}
	return false
}

func describeBashAllowlist(command string) string {
	prefix := extractBashPrefix(command)
	if prefix == "" {
		return ""
	}

	colonIndex := strings.Index(prefix, ":")
	if colonIndex == -1 {
		return ""
	}

	cmdName := prefix[:colonIndex]
	dirPath := prefix[colonIndex+1:]
	if dirPath != "./" {
		return fmt.Sprintf("%s in %s directory (includes subdirs)", cmdName, dirPath)
	}
	return fmt.Sprintf("%s in %s directory", cmdName, dirPath)
}

// extractBashPrefix extracts a prefix pattern from a bash command.
// For commands like "cat tools/tools_test.go | head -200", returns "cat:tools/"
// For commands without path args, returns empty string.
// Paths with ".." traversal that escape the base directory return empty string for security.
func extractBashPrefix(command string) string {
	parts := strings.Split(command, "|")
	firstCommand := strings.TrimSpace(parts[0])

	fields := strings.Fields(firstCommand)
	if len(fields) < 2 {
		return ""
	}

	baseCommand := fields[0]
	safeCommands := map[string]bool{
		"cat": true, "ls": true, "head": true, "tail": true,
		"less": true, "more": true, "file": true, "wc": true,
		"grep": true, "find": true, "tree": true, "stat": true,
		"sed": true,
	}

	if !safeCommands[baseCommand] {
		return ""
	}

	for _, arg := range fields[1:] {
		if strings.HasPrefix(arg, "-") || isNumeric(arg) {
			continue
		}
		if !strings.Contains(arg, "/") && !strings.Contains(arg, "\\") && !strings.HasPrefix(arg, ".") {
			continue
		}

		arg = strings.ReplaceAll(arg, "\\", "/")
		if path.IsAbs(arg) {
			return ""
		}

		cleaned := path.Clean(arg)
		if strings.HasPrefix(cleaned, "..") {
			return ""
		}

		if strings.Contains(arg, "..") {
			originalBase := strings.SplitN(arg, "/", 2)[0]
			cleanedBase := strings.SplitN(cleaned, "/", 2)[0]
			if originalBase != cleanedBase {
				return ""
			}
		}

		isDir := strings.HasSuffix(arg, "/")
		dir := path.Dir(cleaned)
		if isDir {
			dir = cleaned
		}

		if dir == "." {
			return fmt.Sprintf("%s:./", baseCommand)
		}
		return fmt.Sprintf("%s:%s/", baseCommand, dir)
	}

	for _, arg := range fields[1:] {
		if strings.HasPrefix(arg, "-") || isNumeric(arg) {
			continue
		}
		return fmt.Sprintf("%s:./", baseCommand)
	}

	return ""
}

func isNumeric(s string) bool {
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return len(s) > 0
}

// isCommandOutsideCwd checks if a bash command targets paths outside the current working directory.
func isCommandOutsideCwd(command string) bool {
	cwd, err := os.Getwd()
	if err != nil {
		return false
	}

	parts := strings.FieldsFunc(command, func(r rune) bool {
		return r == '|' || r == ';' || r == '&'
	})

	for _, part := range parts {
		fields := strings.Fields(strings.TrimSpace(part))
		if len(fields) == 0 {
			continue
		}

		for _, arg := range fields[1:] {
			if strings.HasPrefix(arg, "-") {
				continue
			}
			if strings.HasPrefix(arg, "/") || strings.HasPrefix(arg, "\\") {
				return true
			}
			if filepath.IsAbs(arg) {
				absPath := filepath.Clean(arg)
				if !strings.HasPrefix(absPath, cwd) {
					return true
				}
				continue
			}
			if strings.HasPrefix(arg, "..") {
				absPath := filepath.Clean(filepath.Join(cwd, arg))
				if !strings.HasPrefix(absPath, cwd) {
					return true
				}
			}
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

func (a *ApprovalManager) matchesHierarchicalPrefix(currentPrefix string) bool {
	colonIndex := strings.Index(currentPrefix, ":")
	if colonIndex == -1 {
		return false
	}

	currentCommand := currentPrefix[:colonIndex]
	currentPath := currentPrefix[colonIndex+1:]

	for storedPrefix := range a.prefixes {
		storedColonIndex := strings.Index(storedPrefix, ":")
		if storedColonIndex == -1 {
			continue
		}

		storedCommand := storedPrefix[:storedColonIndex]
		storedPath := storedPrefix[storedColonIndex+1:]
		if currentCommand == storedCommand && strings.HasPrefix(currentPath, storedPath) {
			return true
		}
	}

	return false
}
