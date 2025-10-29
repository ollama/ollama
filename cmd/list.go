package cmd

import (
	"os"
	"strings"

	"github.com/olekukonko/tablewriter"
	"github.com/ollama/ollama/client"
	"github.com/ollama/ollama/format"
	"github.com/spf13/cobra"
)

func cmdList() *cobra.Command {
	return &cobra.Command{
		Use:     "list [pattern]",
		Aliases: []string{"ls"},
		Short:   "List available models in the local repository",
		Args:    cobra.MaximumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    listHandler,
	}
}

func listHandler(cmd *cobra.Command, args []string) error {
	c := client.New()
	w, err := c.List(cmd.Context())
	if err != nil {
		return err
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"NAME", "ID", "SIZE", "MODIFIED"})
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding("    ")

	for _, m := range w.Models {
		if len(args) == 0 || strings.HasPrefix(strings.ToLower(m.Name), strings.ToLower(args[0])) {
			size := format.HumanBytes(m.Size)
			if m.RemoteModel != "" {
				size = "-"
			}
			table.Append([]string{
				m.Model,
				m.Digest[:12],
				size,
				format.HumanTime(m.ModifiedAt, "Never"),
			})
		}
	}

	table.Render()

	return nil
}
