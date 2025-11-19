package cmd

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/config"
)

// ConfigHandler handles the 'ollama config' command for viewing and setting configuration values
func ConfigHandler(cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		cfg, err := config.Load()
		if err != nil {
			return err
		}
		
		serverURL := cfg.ServerURL
		if serverURL == "" {
			serverURL = "http://localhost:11434"
		}
		fmt.Printf("server_url: %s\n", serverURL)
		return nil
	}

	if len(args) != 2 {
		return fmt.Errorf("usage: ollama config <key> <value>")
	}

	key := args[0]
	// Sanitize input by trimming whitespace
	value := strings.TrimSpace(args[1])

	if key != "server_url" {
		return fmt.Errorf("unknown config key: %s", key)
	}

	// Validate server URL format and security
	if err := validateServerURL(value); err != nil {
		return err
	}

	cfg, err := config.Load()
	if err != nil {
		return err
	}

	cfg.ServerURL = value
	if err := cfg.Save(); err != nil {
		return err
	}

	fmt.Printf("Set %s to %s\n", key, value)
	return nil
}

// validateServerURL validates server URL format and security requirements
func validateServerURL(urlStr string) error {
	parsed, err := url.Parse(urlStr)
	if err != nil {
		return fmt.Errorf("invalid URL format: %v", err)
	}

	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return fmt.Errorf("URL must use http or https scheme")
	}

	if parsed.Host == "" {
		return fmt.Errorf("URL must include a host")
	}

	return nil
}
