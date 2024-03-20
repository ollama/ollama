package cmd

import (
	"fmt"
	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/version"
	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:           "ollama",
	Short:         "Large language model runner",
	SilenceUsage:  true,
	SilenceErrors: true,
	CompletionOptions: cobra.CompletionOptions{
		DisableDefaultCmd: true,
	},
	Run: func(cmd *cobra.Command, args []string) {
		if version, _ := cmd.Flags().GetBool("version"); version {
			versionHandler(cmd, args)
			return
		}

		cmd.Print(cmd.UsageString())
	},
}

func init() {
	rootCmd.Flags().BoolP("version", "v", false, "Show version information")
}

func versionHandler(cmd *cobra.Command, _ []string) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return
	}

	serverVersion, err := client.Version(cmd.Context())
	if err != nil {
		fmt.Println("Warning: could not connect to a running Ollama instance")
	}

	if serverVersion != "" {
		fmt.Printf("ollama version is %s\n", serverVersion)
	}

	if serverVersion != version.Version {
		fmt.Printf("Warning: client version is %s\n", version.Version)
	}
}
