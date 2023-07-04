package cmd

import (
	"context"
	"log"
	"net"
	"net/http"
	"os"
	"path"
	"time"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/server"
	"github.com/spf13/cobra"
)

func sockpath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	return path.Join(home, ".ollama", "ollama.sock")
}

func running() bool {
	// Set a timeout duration
	timeout := time.Second
	// Dial the unix socket
	conn, err := net.DialTimeout("unix", sockpath(), timeout)
	if err != nil {
		return false
	}

	if conn != nil {
		defer conn.Close()
	}
	return true
}

func serve() error {
	sp := sockpath()

	if err := os.MkdirAll(path.Dir(sp), 0o700); err != nil {
		return err
	}

	if err := os.RemoveAll(sp); err != nil {
		return err
	}

	ln, err := net.Listen("unix", sp)
	if err != nil {
		return err
	}

	if err := os.Chmod(sp, 0o700); err != nil {
		return err
	}

	return server.Serve(ln)
}

func NewAPIClient() (*api.Client, error) {
	var err error

	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	socket := path.Join(home, ".ollama", "ollama.sock")

	dialer := &net.Dialer{
		Timeout: 10 * time.Second,
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

	cobra.EnableCommandSorting = false

	runCmd := &cobra.Command{
		Use: "run MODEL",
		Short: "Run a model",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command,args []string) error {
			return nil
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
