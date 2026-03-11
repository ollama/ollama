package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
)

// Tier constants mirror fitcheck.CompatibilityTier values.
const (
	fitTierIdeal    = 0
	fitTierGood     = 1
	fitTierMarginal = 2
	fitTierPossible = 3
	fitTierTooLarge = 4
)

// RunFit is the cobra RunE handler for the "ollama fit" command.
func RunFit(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	showAll, _ := cmd.Flags().GetBool("all")
	family, _ := cmd.Flags().GetString("family")
	tags, _ := cmd.Flags().GetString("tags")
	jsonOut, _ := cmd.Flags().GetBool("json")

	resp, err := client.Fit(cmd.Context(), api.FitRequest{
		All:    showAll,
		Family: family,
		Tags:   tags,
	})
	if err != nil {
		return err
	}

	if jsonOut {
		return json.NewEncoder(os.Stdout).Encode(resp)
	}

	renderFitTTY(*resp)
	return nil
}

func renderFitTTY(resp api.FitResponse) {
	hw := resp.System

	// Header: system info
	fmt.Println("Ollama Fit Check")
	fmt.Println(strings.Repeat("\u2500", 62))
	fmt.Printf("  CPU  : %s (%s)\n", hw.OS, hw.Arch)
	fmt.Printf("  RAM  : %s free / %s total\n",
		fitHumanizeBytes(hw.RAMAvailableBytes), fitHumanizeBytes(hw.RAMTotalBytes))
	if hw.BestGPU != nil {
		fmt.Printf("  GPU  : %s %s  \u2022  %s free / %s total\n",
			hw.BestGPU.Library, hw.BestGPU.Name,
			fitHumanizeBytes(hw.BestGPU.FreeMemory), fitHumanizeBytes(hw.BestGPU.TotalMemory))
	} else {
		fmt.Println("  GPU  : None detected")
	}
	fmt.Printf("  Disk : %s free  \u2192  %s\n",
		fitHumanizeBytes(hw.DiskModelAvailBytes), hw.ModelsDir)
	fmt.Println(strings.Repeat("\u2500", 62))
	fmt.Println()

	// Group by tier (tier is an int matching fitcheck.CompatibilityTier iota values)
	groups := map[int][]api.FitModelCandidate{}
	for _, c := range resp.Models {
		groups[c.Tier] = append(groups[c.Tier], c)
	}

	tiers := []struct {
		tier  int
		icon  string
		label string
	}{
		{fitTierIdeal, "\u2705", "IDEAL \u2014 Full GPU inference, fast"},
		{fitTierGood, "\U0001f7e1", "GOOD \u2014 Minor CPU offload"},
		{fitTierMarginal, "\U0001f7e0", "MARGINAL \u2014 Significant CPU offload, slow"},
		{fitTierPossible, "\u2b1c", "POSSIBLE \u2014 CPU only, very slow"},
		{fitTierTooLarge, "\U0001f534", "TOO LARGE \u2014 Cannot run on this hardware"},
	}

	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	for _, tg := range tiers {
		entries, ok := groups[tg.tier]
		if !ok || len(entries) == 0 {
			continue
		}
		fmt.Printf("  %s  %s\n", tg.icon, tg.label)
		fmt.Println("  " + strings.Repeat("\u2500", 60))
		for _, e := range entries {
			diskStr := fitHumanizeBytes(e.Req.DiskSizeMB * 1024 * 1024)
			if e.Installed {
				diskStr = "installed"
			}
			tpsStr := fmt.Sprintf("~%d tok/s", e.EstTPS)
			nameStr := e.Req.Name
			if e.Installed {
				nameStr += " ✓"
			}
			fmt.Fprintf(tw, "  %-28s\t%-8s\t%8s\t%12s\t%s\n",
				nameStr, e.Req.Quant, diskStr, tpsStr, e.RunMode)
			for _, note := range e.Notes {
				fmt.Fprintf(tw, "       \u26a0  %s\n", note)
			}
		}
		tw.Flush()
		fmt.Println()
	}
}

// fitHumanizeBytes formats byte counts in human-readable units.
func fitHumanizeBytes(b uint64) string {
	const (
		kb  = uint64(1024)
		mb  = uint64(1024 * 1024)
		_gb = uint64(1024 * 1024 * 1024)
	)
	switch {
	case b >= _gb:
		return fmt.Sprintf("%.1f GB", float64(b)/float64(_gb))
	case b >= mb:
		return fmt.Sprintf("%.1f MB", float64(b)/float64(mb))
	case b >= kb:
		return fmt.Sprintf("%d KB", b/kb)
	default:
		return fmt.Sprintf("%d B", b)
	}
}

// fitCmd returns the cobra command for "ollama fit".
func fitCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "fit",
		Short: "Check which models are compatible with this machine",
		Long:  "Scans your hardware and ranks available models by how well they will run on your system.",
		RunE:  RunFit,
	}
	cmd.Flags().Bool("all", false, "Show all models including ones too large to run")
	cmd.Flags().String("family", "", "Filter to a specific model family (e.g. llama3, mistral)")
	cmd.Flags().String("tags", "", "Filter by tag (e.g. code, vision, embed)")
	cmd.Flags().Bool("json", false, "Output raw JSON")
	return cmd
}
