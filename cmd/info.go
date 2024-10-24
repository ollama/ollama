package cmd

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

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
			indent, "Filesystem Used:", format.HumanBytes2(resp.Models.FilesystemUsed),
		},
		{
			indent, "Running:", strconv.Itoa(resp.Models.Running),
		},
		{
			indent, "VRAM Used:", format.HumanBytes2(resp.Models.VRAMUsed),
		},
	}
	fmt.Fprintf(out, "%sModels:\n", indent)
	table.AppendBulk(data)
	table.Render()
}

func prettyPrintCompute(out io.Writer, indent string, resp api.InfoResponse) {
	table := tablewriter.NewWriter(os.Stdout)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding(" ")
	data := [][]string{
		{
			indent, "Available Runners:", strings.Join(resp.ComputeInfo.AvailableRunners, ", "),
		},
	}
	fmt.Fprintf(out, "%sCompute:\n", indent)
	table.AppendBulk(data)
	table.Render()
	indent += " "
	prettyPrintSystem(out, indent, resp)
	prettyPrintSupportedGPUs(out, indent, resp)
	prettyPrintUnsupportedGPUs(out, indent, resp)
	na := ""
	if len(resp.ComputeInfo.DiscoveryErrors) == 0 {
		na = " N/A"
	}
	fmt.Fprintf(out, "%sDiscovery Errors:%s\n", indent, na)
	indent += " "
	for _, msg := range resp.ComputeInfo.DiscoveryErrors {
		fmt.Fprintf(out, "%s%s\n", indent, msg)
	}
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
			indent, "Total Memory:", format.HumanBytes2(resp.ComputeInfo.SystemCompute.TotalMemory),
		},
		{
			indent, "Free Memory:", format.HumanBytes2(resp.ComputeInfo.SystemCompute.FreeMemory),
		},
		{
			indent, "Free Swap:", format.HumanBytes2(resp.ComputeInfo.SystemCompute.FreeSwap),
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
			indent, "Total Memory:", format.HumanBytes2(gpu.TotalMemory),
		},
		{
			indent, "Free Memory:", format.HumanBytes2(gpu.FreeMemory),
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

func prettyPrintUnsupportedGPUs(out io.Writer, indent string, resp api.InfoResponse) {
	na := ""
	if len(resp.ComputeInfo.UnsupportedGPUs) == 0 {
		na = " N/A"
	}
	fmt.Fprintf(out, "%sUnsupported GPUs:%s\n", indent, na)
	indent += " "
	for _, gpu := range resp.ComputeInfo.UnsupportedGPUs {
		prettyPrintUnsupportedGPU(out, indent, gpu)
	}
}

func prettyPrintUnsupportedGPU(out io.Writer, indent string, gpu api.UnsupportedGPUInfo) {
	table := tablewriter.NewWriter(os.Stdout)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding(" ")
	data := [][]string{
		{
			indent, "Error:", gpu.Error,
		},
		{
			indent, "Name:", gpu.Name,
		},
		{
			indent, "Total Memory:", format.HumanBytes2(gpu.TotalMemory),
		},
		{
			indent, "Free Memory:", format.HumanBytes2(gpu.FreeMemory),
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
