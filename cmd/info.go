package cmd

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"

	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/version"
)

func InfoHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	resp, err := client.Info(cmd.Context())
	if err != nil {
		return err
	}
	out := os.Stdout
	prettyPrintClientInfo(out)
	fmt.Fprint(out, "\n")

	prettyPrintInfoResponse(out, *resp)

	return nil
}

func prettyPrintClientInfo(out io.Writer) {
	table := tablewriter.NewWriter(os.Stdout)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding(" ")
	indent := ""

	cfgDir := ""
	home, err := os.UserHomeDir()
	if err == nil {
		cfgDir = filepath.Join(home, ".ollama")
	}

	data := [][]string{
		{
			indent, "Version:", version.Version,
		},
		{
			indent, "Configuration:", cfgDir,
		},
		{
			indent, "Connection:", envconfig.Host().String(),
		},
	}
	fmt.Fprint(out, "Client:\n")
	table.AppendBulk(data)
	table.Render()
}

func prettyPrintInfoResponse(out io.Writer, resp api.InfoResponse) {
	table := tablewriter.NewWriter(os.Stdout)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding(" ")
	indent := ""
	data := [][]string{
		{
			indent, "Version:", resp.Version,
		},
	}
	fmt.Fprint(out, "Server:\n")
	table.AppendBulk(data)
	table.Render()
	prettyPrintModels(out, " ", resp)
	prettyPrintCompute(out, " ", resp)
}

func prettyPrintModels(out io.Writer, indent string, resp api.InfoResponse) {
	table := tablewriter.NewWriter(os.Stdout)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding(" ")
	data := [][]string{
		{
			indent, "Store:", envconfig.Models(),
		},
		{
			indent, "Downloaded:", strconv.Itoa(resp.Models.Count),
		},
		{
			indent, "Filesystem Used:", format.HumanBytes(int64(resp.Models.FilesystemUsed)),
		},
		{
			indent, "Running:", strconv.Itoa(resp.Models.Running),
		},
		{
			indent, "VRAM Used:", format.HumanBytes(int64(resp.Models.VRAMUsed)),
		},
	}
	fmt.Fprintf(out, "%sModels:\n", indent)
	table.AppendBulk(data)
	table.Render()
}

func prettyPrintCompute(out io.Writer, indent string, resp api.InfoResponse) {
	fmt.Fprintf(out, "%sCompute:\n", indent)
	indent += " "
	prettyPrintSystem(out, indent, resp)
	prettyPrintSupportedGPUs(out, indent, resp)
}

func prettyPrintSystem(out io.Writer, indent string, resp api.InfoResponse) {
	table := tablewriter.NewWriter(os.Stdout)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding(" ")
	data := [][]string{
		{
			indent, "CPU Cores:", strconv.Itoa(resp.ComputeInfo.SystemCompute.CPUCores),
		},
		{
			indent, "Total Memory:", format.HumanBytes(int64(resp.ComputeInfo.SystemCompute.TotalMemory)),
		},
		{
			indent, "Free Memory:", format.HumanBytes(int64(resp.ComputeInfo.SystemCompute.FreeMemory)),
		},
		{
			indent, "Free Swap:", format.HumanBytes(int64(resp.ComputeInfo.SystemCompute.FreeSwap)),
		},
	}
	fmt.Fprintf(out, "%sSystem:\n", indent)
	table.AppendBulk(data)
	table.Render()
}

func prettyPrintSupportedGPUs(out io.Writer, indent string, resp api.InfoResponse) {
	fmt.Fprintf(out, "%sSupported GPUs:\n", indent)
	indent += " "
	for _, gpu := range resp.ComputeInfo.SupportedGPUs {
		prettyPrintSupportedGPU(out, indent, gpu)
	}
}

func prettyPrintSupportedGPU(out io.Writer, indent string, gpu api.GPUInfo) {
	table := tablewriter.NewWriter(os.Stdout)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding(" ")
	data := [][]string{
		{
			indent, "Name:", gpu.Name,
		},
		{
			indent, "Total Memory:", format.HumanBytes(int64(gpu.TotalMemory)),
		},
		{
			indent, "Free Memory:", format.HumanBytes(int64(gpu.FreeMemory)),
		},
		{
			indent, "Compute:", gpu.Compute,
		},
		{
			indent, "Driver:", gpu.Driver,
		},
	}
	fmt.Fprintf(out, "%s%s %s:\n", indent, gpu.Runner, gpu.ID)
	table.AppendBulk(data)
	table.Render()
}
