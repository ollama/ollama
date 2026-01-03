package server

import (
	"path/filepath"
	"strings"
)

// =============================================================================
// MCP Security Configuration
// =============================================================================
//
// SECURITY REVIEW: This file defines the security policies that control which
// commands can be executed as MCP servers and what arguments/environment they
// can receive. Changes to this file should be reviewed by security-aware
// maintainers.
//
// Key security surfaces:
//   - BlockedCommands: Prevents execution of dangerous system commands
//   - BlockedMetacharacters: Prevents shell injection attacks
//   - FilteredEnvironmentVars: Prevents credential leakage to MCP servers
//
// Threat model:
//   - Malicious MCP server configs attempting to execute system commands
//   - Shell injection through tool arguments
//   - Credential theft through environment variable access
//
// =============================================================================

// MCPSecurityConfig defines security policies for MCP servers
type MCPSecurityConfig struct {
	// Commands that are never allowed as MCP servers
	BlockedCommands []string

	// Shell metacharacters that are not allowed in arguments
	BlockedMetacharacters []string

	// Environment variables that should be filtered
	FilteredEnvironmentVars []string
}

// DefaultSecurityConfig returns the default security configuration.
//
// SECURITY REVIEW: This function defines the default blocklists. Adding or
// removing entries has direct security implications. Consider:
//   - Why is a command being added/removed?
//   - What attack vectors does it enable/prevent?
//   - Are there bypass possibilities (symlinks, PATH manipulation)?
func DefaultSecurityConfig() *MCPSecurityConfig {
	return &MCPSecurityConfig{
		// SECURITY: Blocked commands - these can never be used as MCP server commands.
		// Rationale: These commands could be used for privilege escalation,
		// arbitrary file manipulation, or establishing network connections.
		BlockedCommands: []string{
			// Shells - prevent arbitrary command execution
			"sh", "bash", "zsh", "fish", "csh", "ksh", "dash", "tcsh",
			"cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe",

			// System commands - prevent privilege escalation and system damage
			"sudo", "su", "doas", "runas", "pkexec",
			"rm", "del", "rmdir", "format", "dd", "shred",
			"kill", "killall", "pkill", "shutdown", "reboot",
			"systemctl", "service", "init",

			// Network tools - prevent data exfiltration and network attacks
			"curl", "wget", "nc", "netcat", "telnet", "ssh", "scp", "sftp",
			"nmap", "ping", "traceroute", "dig", "nslookup",

			// Script interpreters - prevent arbitrary code execution
			"eval", "exec", "source", ".",
			"perl", "ruby", "php", "lua", "tcl",

			// File manipulation - prevent permission/ownership changes
			"chmod", "chown", "chgrp", "mount", "umount",
			"ln", "mkfifo", "mknod",

			// Package managers - prevent system modification
			"apt", "apt-get", "yum", "dnf", "pacman", "zypper",
			"brew", "port", "snap", "flatpak",
		},
		
		// SECURITY: Shell metacharacters - block these in arguments to prevent injection.
		// These characters could be used to chain commands or redirect I/O.
		BlockedMetacharacters: []string{
			";", "|", "&", "$(", "`", ">", "<", ">>", "<<",
			"||", "&&", "\n", "\r", "$", "!", "*", "?", "=",
		},

		// SECURITY: Environment variables to filter - prevent credential leakage.
		// MCP servers should not have access to credentials in the parent environment.
		FilteredEnvironmentVars: []string{
			// AWS
			"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
			
			// Cloud providers
			"GOOGLE_APPLICATION_CREDENTIALS", "AZURE_CLIENT_SECRET",
			
			// API Keys
			"GITHUB_TOKEN", "GITLAB_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
			
			// Database
			"DATABASE_URL", "DB_PASSWORD", "MYSQL_ROOT_PASSWORD", "POSTGRES_PASSWORD",
			
			// Authentication
			"JWT_SECRET", "SESSION_SECRET", "AUTH_TOKEN", "API_KEY", "API_SECRET",
			
			// SSH
			"SSH_AUTH_SOCK", "SSH_AGENT_PID",
		},
	}
}

// IsCommandAllowed checks if a command is allowed by security policy
func (c *MCPSecurityConfig) IsCommandAllowed(command string) bool {
	baseName := filepath.Base(command)

	for _, blocked := range c.BlockedCommands {
		if baseName == blocked || strings.HasSuffix(command, "/"+blocked) {
			return false
		}
	}

	return true
}

// HasShellMetacharacters checks if a string contains shell metacharacters
func (c *MCPSecurityConfig) HasShellMetacharacters(s string) bool {
	for _, meta := range c.BlockedMetacharacters {
		if strings.Contains(s, meta) {
			return true
		}
	}
	return false
}

// ShouldFilterEnvironmentVar checks if an environment variable should be filtered
func (c *MCPSecurityConfig) ShouldFilterEnvironmentVar(key string) bool {
	for _, filtered := range c.FilteredEnvironmentVars {
		if key == filtered {
			return true
		}
	}
	return false
}

// Global security config instance
// NOTE: To add user customization, load from ~/.ollama/mcp-security.json and append to these defaults
var globalSecurityConfig = DefaultSecurityConfig()

// GetSecurityConfig returns the global security configuration
func GetSecurityConfig() *MCPSecurityConfig {
	return globalSecurityConfig
}