package cmd

import (
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/olekukonko/tablewriter"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/spf13/cobra"
)

func NewPsCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "ps",
		Short:   "List running models",
		PreRunE: checkServerHeartbeat,
		RunE:    listRunningHandler,
	}

	return cmd
}

func listRunningHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	models, err := client.ListRunning(cmd.Context())
	if err != nil {
		return err
	}

	var data [][]string

	for _, m := range models.Models {
		if len(args) == 0 || strings.HasPrefix(m.Name, args[0]) {
			var procStr string
			switch {
			case m.SizeVRAM == 0:
				procStr = "100% CPU"
			case m.SizeVRAM == m.Size:
				procStr = "100% GPU"
			case m.SizeVRAM > m.Size || m.Size == 0:
				procStr = "Unknown"
			default:
				sizeCPU := m.Size - m.SizeVRAM
				cpuPercent := math.Round(float64(sizeCPU) / float64(m.Size) * 100)
				procStr = fmt.Sprintf("%d%%/%d%% CPU/GPU", int(cpuPercent), int(100-cpuPercent))
			}

			var until string
			delta := time.Since(m.ExpiresAt)
			if delta > 0 {
				until = "Stopping..."
			} else {
				until = format.HumanTime(m.ExpiresAt, "Never")
			}
			data = append(data, []string{m.Name, m.Digest[:12], format.HumanBytes(m.Size), procStr, until})
		}
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"NAME", "ID", "SIZE", "PROCESSOR", "UNTIL"})
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding("    ")
	table.AppendBulk(data)
	table.Render()

	return nil
}
