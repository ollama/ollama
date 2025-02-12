package cmd

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
)

func NewStopCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "stop MODEL",
		Short:   "Stop a running model",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    stopHandler,
	}

	return cmd
}

func stopHandler(cmd *cobra.Command, args []string) error {
	opts := &runOptions{
		Model:     args[0],
		KeepAlive: &api.Duration{Duration: 0},
	}
	if err := loadOrUnloadModel(cmd, opts); err != nil {
		if strings.Contains(err.Error(), "not found") {
			return fmt.Errorf("couldn't find model \"%s\" to stop", args[0])
		}
	}
	return nil
}
