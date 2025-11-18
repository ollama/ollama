package cmd

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/config"
)

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
	value := args[1]

	if key != "server_url" {
		return fmt.Errorf("unknown config key: %s", key)
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
