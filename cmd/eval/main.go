package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

func main() {
	model := flag.String("model", "", "model to evaluate")
	suite := flag.String("suite", "", "comma-separated list of suites to run (empty runs all)")
	list := flag.Bool("list", false, "list available suites")
	verbose := flag.Bool("v", false, "verbose output")
	timeout := flag.Int("timeout", 60, "timeout per test in seconds")
	export := flag.String("export", "eval-results.json", "export results to file")
	flag.Parse()

	if *list {
		for _, s := range suites {
			fmt.Printf("%s (%d tests)\n", s.Name, len(s.Tests))
		}
		return
	}

	if *model == "" {
		fmt.Fprintf(os.Stderr, "error: -model parameter is required\n")
		os.Exit(1)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	if err := client.Heartbeat(ctx); err != nil {
		cancel()
		fmt.Fprintf(os.Stderr, "error: cannot connect to ollama\n")
		os.Exit(1)
	}
	cancel()

	selected := suites
	if *suite != "" {
		suiteNames := strings.Split(*suite, ",")
		selected = []Suite{}
		var notFound []string

		for _, name := range suiteNames {
			name = strings.TrimSpace(name)
			if name == "" {
				continue
			}

			found := false
			for _, s := range suites {
				if s.Name == name {
					selected = append(selected, s)
					found = true
					break
				}
			}
			if !found {
				notFound = append(notFound, name)
			}
		}

		if len(notFound) > 0 {
			fmt.Fprintf(os.Stderr, "error: suite(s) not found: %s\n", strings.Join(notFound, ", "))
			os.Exit(1)
		}
	}

	var results []Result
	for _, s := range selected {
		if *verbose {
			fmt.Printf("\n%s (%d tests)\n", s.Name, len(s.Tests))
		}
		for i, test := range s.Tests {
			if test.Options == nil {
				test.Options = map[string]any{"temperature": 0.1}
			}
			if test.Check == nil {
				test.Check = HasResponse()
			}

			if *verbose {
				fmt.Printf("  [%d/%d] %s... ", i+1, len(s.Tests), test.Name)
			}

			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeout)*time.Second)
			result := Run(ctx, client, *model, test)
			cancel()

			results = append(results, result)

			if *verbose {
				if result.Error != nil {
					fmt.Printf("ERROR: %v\n", result.Error)
				} else if result.Passed {
					fmt.Printf("PASS (%.2fs)", result.Duration.Seconds())
					if len(result.Tools) > 0 || result.Thinking {
						fmt.Printf(" [")
						if len(result.Tools) > 0 {
							fmt.Printf("tools: %s", strings.Join(result.Tools, ","))
						}
						if result.Thinking {
							if len(result.Tools) > 0 {
								fmt.Printf(", ")
							}
							fmt.Printf("thinking")
						}
						fmt.Printf("]")
					}
					fmt.Println()

					// Print tool calls with details
					if len(result.ToolCalls) > 0 {
						fmt.Printf("    Tool Calls:\n")
						for _, tc := range result.ToolCalls {
							argsJSON, _ := json.Marshal(tc.Function.Arguments)
							fmt.Printf("      - %s: %s\n", tc.Function.Name, string(argsJSON))
						}
					}

					// Print response if there is one
					if result.Response != "" {
						fmt.Printf("    Response: %s\n", result.Response)
					}
				} else {
					fmt.Printf("FAIL (%.2fs)\n", result.Duration.Seconds())

					// Print tool calls with details even on failure
					if len(result.ToolCalls) > 0 {
						fmt.Printf("    Tool Calls:\n")
						for _, tc := range result.ToolCalls {
							argsJSON, _ := json.Marshal(tc.Function.Arguments)
							fmt.Printf("      - %s: %s\n", tc.Function.Name, string(argsJSON))
						}
					}

					// Print response even on failure
					if result.Response != "" {
						fmt.Printf("    Response: %s\n", result.Response)
					}
				}
			}
		}
	}

	printSummary(results)

	if *export != "" {
		if err := writeJSON(*export, results); err != nil {
			fmt.Fprintf(os.Stderr, "warning: export failed: %v\n", err)
		} else if *verbose {
			fmt.Printf("\nResults: %s\n", *export)
		}
	}

	if anyFailed(results) {
		os.Exit(1)
	}
}

func printSummary(results []Result) {
	var passed, failed, errors int
	for _, r := range results {
		if r.Error != nil {
			errors++
		} else if r.Passed {
			passed++
		} else {
			failed++
		}
	}

	total := len(results)
	rate := 0.0
	if total > 0 {
		rate = float64(passed) / float64(total) * 100
	}

	fmt.Printf("\n%d/%d passed (%.1f%%)", passed, total, rate)
	if errors > 0 {
		fmt.Printf(", %d errors", errors)
	}
	fmt.Println()
}

func anyFailed(results []Result) bool {
	for _, r := range results {
		if !r.Passed || r.Error != nil {
			return true
		}
	}
	return false
}

func writeJSON(path string, results []Result) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(results)
}
