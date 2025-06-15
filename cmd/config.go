package cmd

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/envconfig"
)

func ConfigCommand() *cobra.Command {
	configCmd := &cobra.Command{
		Use:   "config",
		Short: "Manage Ollama configuration",
		Long:  "Manage Ollama configuration through config file and environment variables",
	}

	showCmd := &cobra.Command{
		Use:   "show",
		Short: "Show current configuration",
		RunE:  configShowHandler,
	}

	initCmd := &cobra.Command{
		Use:   "init",
		Short: "Initialize configuration file",
		RunE:  configInitHandler,
	}

	configCmd.AddCommand(showCmd, initCmd)
	return configCmd
}

func configShowHandler(cmd *cobra.Command, args []string) error {
	// Get all environment variables
	envVars := envconfig.Values()
	
	// Print configuration sources
	fmt.Println("Configuration Sources:")
	fmt.Println("--------------------")
	
	// Check if config file is loaded
	paths := envconfig.GetConfigPaths()
	for _, path := range paths {
		if _, err := os.Stat(path); err == nil {
			fmt.Printf("Config file: %s\n", path)
		}
	}
	
	fmt.Println("\nCurrent Configuration:")
	fmt.Println("--------------------")
	
	// Print all configuration values
	for key, value := range envVars {
		if value != "" {
			fmt.Printf("%s = %s\n", key, value)
		}
	}
	
	return nil
}

func configInitHandler(cmd *cobra.Command, args []string) error {
	// Get the first config path for the current OS
	paths := envconfig.GetConfigPaths()
	if len(paths) == 0 {
		return fmt.Errorf("no valid config paths found for current OS")
	}
	
	configPath := paths[0]
	
	// Create directory if it doesn't exist
	configDir := filepath.Dir(configPath)
	if err := os.MkdirAll(configDir, 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}
	
	// Check if file already exists
	if _, err := os.Stat(configPath); err == nil {
		return fmt.Errorf("config file already exists at %s", configPath)
	}
	
	// Generate example config
	exampleConfig := envconfig.GenerateExampleConfig()
	
	// Write config file
	if err := os.WriteFile(configPath, []byte(exampleConfig), 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}
	
	fmt.Printf("Created example configuration file at: %s\n", configPath)
	fmt.Println("Please edit the file to customize your configuration.")
	
	return nil
} 