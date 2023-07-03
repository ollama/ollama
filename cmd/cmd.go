package cmd

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"path"
	"time"

	"github.com/spf13/cobra"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/server"
)

func NewAPIClient(cmd *cobra.Command) (*api.Client, error) {
	var rawKey []byte
	var err error

	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	socket := path.Join(home, ".ollama", "ollama.sock")

	dialer := &net.Dialer{
		Timeout: 10 * time.Second,
	}

	k, _ := cmd.Flags().GetString("key")

	if k != "" {
		fn := path.Join(home, ".ollama/keys/", k)
		rawKey, err = os.ReadFile(fn)
		if err != nil {
			return nil, err
		}
	}

	return &api.Client{
		URL: "http://localhost",
		HTTP: http.Client{
			Transport: &http.Transport{
				DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
					return dialer.DialContext(ctx, "unix", socket)
				},
			},
		},
		PrivateKey: rawKey,
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
		},
	}

	rootCmd.PersistentFlags().StringP("key", "k", "", "Private key to use for authenticating")

	cobra.EnableCommandSorting = false

	modelsCmd := &cobra.Command{
		Use:   "models",
		Args:  cobra.MaximumNArgs(1),
		Short: "List models",
		Long:  "List the models",
		RunE: func(cmd *cobra.Command, args []string) error {
			client, err := NewAPIClient(cmd)
			if err != nil {
				return err
			}
			fmt.Printf("client = %q\n", client)
			return nil
		},
	}

	runCmd := &cobra.Command{
		Use: "run",
		Short: "Run a model and submit prompts.",
		RunE: func(cmd *cobra.Command,args []string) error {
			return nil
		},
	}

	serveCmd := &cobra.Command{
		Use:     "serve",
		Aliases: []string{"start"},
		Short:   "Start ollama",
		RunE: func(cmd *cobra.Command, args []string) error {
			home, err := os.UserHomeDir()
			if err != nil {
				return err
			}

			socket := path.Join(home, ".ollama", "ollama.sock")
			if err := os.MkdirAll(path.Dir(socket), 0o700); err != nil {
				return err
			}

			if err := os.RemoveAll(socket); err != nil {
				return err
			}

			ln, err := net.Listen("unix", socket)
			if err != nil {
				return err
			}

			if err := os.Chmod(socket, 0o700); err != nil {
				return err
			}

			return server.Serve(ln)
		},
	}

	rootCmd.AddCommand(
		modelsCmd,
		serveCmd,
		runCmd,
	)

	return rootCmd
}
