package cmd

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"path"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/server"
	"github.com/spf13/cobra"
)

func cacheDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	return path.Join(home, ".ollama")
}

func run(model string) error {
	client, err := NewAPIClient()
	if err != nil {
		return err
	}
	pr := api.PullRequest{
		Model: model,
	}
	callback := func(progress string) {
		fmt.Println(progress)
	}
	_, err = client.Pull(context.Background(), &pr, callback)
	return err
}

func serve() error {
	ln, err := net.Listen("tcp", "127.0.0.1:11434")
	if err != nil {
		return err
	}

	return server.Serve(ln)
}

func NewAPIClient() (*api.Client, error) {
	return &api.Client{
		URL: "http://localhost:11434",
	}, nil
}

func NewCLI() *cobra.Command {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	rootCmd := &cobra.Command{
		Use:   "ollama",
		Short: "Large language model runner",
		CompletionOptions: cobra.CompletionOptions{
			DisableDefaultCmd: true,
		},
		PersistentPreRun: func(cmd *cobra.Command, args []string) {
			// Disable usage printing on errors
			cmd.SilenceUsage = true
			// create the models directory and it's parent
			if err := os.MkdirAll(path.Join(cacheDir(), "models"), 0o700); err != nil {
				panic(err)
			}
		},
	}

	cobra.EnableCommandSorting = false

	runCmd := &cobra.Command{
		Use:   "run MODEL",
		Short: "Run a model",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return run(args[0])
		},
	}

	serveCmd := &cobra.Command{
		Use:     "serve",
		Aliases: []string{"start"},
		Short:   "Start ollama",
		RunE: func(cmd *cobra.Command, args []string) error {
			return serve()
		},
	}

	rootCmd.AddCommand(
		serveCmd,
		runCmd,
	)

	return rootCmd
}
