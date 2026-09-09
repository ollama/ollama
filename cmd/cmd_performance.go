package cmd

import (
	"fmt"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/internal/performance"
)

func init() {
	performanceCmd.Flags().Bool("force", false, "Terminate high memory consuming processes (>40% RAM)")
	performanceCmd.Flags().Bool("optimize", false, "Perform automated memory optimizations (unload idle models) and show recommendations")
}

var performanceCmd = &cobra.Command{
	Use:     "performance",
	Aliases: []string{"/performance"},
	Short:   "Analyze and optimize system memory for Ollama",
	RunE:    PerformanceHandler,
}

func PerformanceHandler(cmd *cobra.Command, args []string) error {
	force, _ := cmd.Flags().GetBool("force")
	optimize, _ := cmd.Flags().GetBool("optimize")

	// 1. Get memory stats
	stats, err := performance.GetMemoryStats()
	if err != nil {
		return fmt.Errorf("failed to retrieve memory stats: %w", err)
	}

	// Print basic memory info
	totalStr := format.HumanBytes(int64(stats.Total))
	requiredStr := format.HumanBytes(int64(float64(stats.Total) * 0.6))

	fmt.Printf("System Memory: %s\n", totalStr)
	fmt.Printf("Required for Ollama: %s (60%%)\n\n", requiredStr)

	// 2. Detect high memory consumers (>40% RAM)
	highMemProcs, err := performance.GetHighMemoryProcesses(40.0, stats.Total)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: Failed to retrieve process list: %v\n", err)
	} else if len(highMemProcs) > 0 {
		fmt.Println("Detected High Memory Consumers:")
		w := tabwriter.NewWriter(os.Stdout, 0, 0, 4, ' ', 0)
		fmt.Fprintln(w, "PID\tProcess\tMemory")
		for _, p := range highMemProcs {
			fmt.Fprintf(w, "%d\t%s\t%.1f%%\n", p.PID, p.Name, p.MemoryPercent)
		}
		w.Flush()
		fmt.Println()

		// If --force is specified, ask to terminate
		if force {
			if promptYesNo("Terminate these processes? (y/N): ") {
				fmt.Println("Terminating processes...")
				for _, p := range highMemProcs {
					if err := performance.KillProcess(p); err != nil {
						fmt.Printf("Error terminating process %s (PID %d): %v\n", p.Name, p.PID, err)
					} else {
						fmt.Printf("✓ Terminated %s (PID %d)\n", p.Name, p.PID)
					}
				}
				fmt.Println()
			} else {
				fmt.Println("Termination cancelled.\n")
			}
		}
	} else {
		fmt.Println("No high memory processes (>40% RAM) detected.\n")
	}

	// 3. Optimize mode
	if optimize {
		fmt.Println("Running optimizations...")
		client, clientErr := api.ClientFromEnvironment()
		if clientErr != nil {
			fmt.Printf("Warning: Could not connect to Ollama server: %v\n\n", clientErr)
		} else {
			optResult, optErr := performance.Optimize(cmd.Context(), client, stats)
			if optErr != nil {
				fmt.Printf("Error running optimizations: %v\n\n", optErr)
			} else {
				if len(optResult.UnloadedModels) > 0 {
					fmt.Println("Unloaded running models:")
					for _, m := range optResult.UnloadedModels {
						fmt.Printf("✓ Model '%s' unloaded\n", m)
					}
					fmt.Println()
				} else {
					fmt.Println("✓ No running models needed to be unloaded.\n")
				}

				fmt.Println("Recommended:")
				for _, rec := range optResult.Recommendations {
					fmt.Printf("✓ %s\n", rec)
				}
				fmt.Println()
			}
		}
	} else {
		// Just show recommendations by default
		totalGB := float64(stats.Total) / (1024 * 1024 * 1024)
		fmt.Println("Memory Optimization Recommendations:")
		if totalGB < 16.0 {
			fmt.Println("- Reduce context window to 2048 or 4096 (e.g. num_ctx=4096)")
			fmt.Println("- Use highly quantized models (e.g., q4_K_M or q3_K_L)")
			fmt.Println("- Enable aggressive model swapping (OLLAMA_NUM_PARALLEL=1)")
		} else if totalGB < 32.0 {
			fmt.Println("- Reduce context window from 32768 to 8192 (e.g. num_ctx=8192)")
			fmt.Println("- Use q4_K_M or q5_K_M quantized models")
			fmt.Println("- Set keep-alive to a lower duration (e.g., --keepalive 5m)")
		} else {
			fmt.Println("- Keep context window at or below 16384 for large models")
			fmt.Println("- Use q4_K_M or q8_0 quantized models for optimal performance")
		}
		fmt.Println("\nRun 'ollama performance --optimize' to unload running models and apply recommendations.")
	}

	return nil
}

func promptYesNo(prompt string) bool {
	fmt.Print(prompt)
	var response string
	_, err := fmt.Scanln(&response)
	if err != nil {
		return false
	}
	response = strings.TrimSpace(strings.ToLower(response))
	return response == "y" || response == "yes"
}
