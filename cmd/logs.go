package cmd

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/app/lifecycle"
)

// LogsHandler handles the logs command
func LogsHandler(cmd *cobra.Command, args []string) error {
	follow, err := cmd.Flags().GetBool("follow")
	if err != nil {
		return err
	}

	tail, err := cmd.Flags().GetInt("tail")
	if err != nil {
		return err
	}

	showApp, err := cmd.Flags().GetBool("app")
	if err != nil {
		return err
	}

	showServer, err := cmd.Flags().GetBool("server")
	if err != nil {
		return err
	}

	// Default to server logs if neither is specified
	if !showApp && !showServer {
		showServer = true
	}

	// Determine which log file to read
	var logFile string
	if showApp {
		logFile = getAppLogFile()
	} else {
		logFile = getServerLogFile()
	}

	// Check if log file exists
	if _, err := os.Stat(logFile); os.IsNotExist(err) {
		if !follow { // Only return error if not in follow mode
			return fmt.Errorf("log file does not exist: %s", logFile)
		}
	}

	if follow {
		return followLogs(cmd.Context(), logFile, tail)
	}

	return showLogs(logFile, tail)
}

// getServerLogFile returns the configured server log file path from the lifecycle package.
func getServerLogFile() string {
	return lifecycle.ServerLogFile
}

// getAppLogFile returns the configured app log file path from the lifecycle package.
func getAppLogFile() string {
	return lifecycle.AppLogFile
}

// showLogs displays the last n lines of the log file
func showLogs(logFile string, tail int) error {
	file, err := os.Open(logFile)
	if err != nil {
		return fmt.Errorf("failed to open log file: %w", err)
	}
	defer file.Close()

	if tail <= 0 {
		// Show all lines
		_, err := io.Copy(os.Stdout, file)
		return err
	}

	// Read all lines and show the last 'tail' lines
	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading log file: %w", err)
	}

	// Show the last 'tail' lines
	start := len(lines) - tail
	if start < 0 {
		start = 0
	}

	for i := start; i < len(lines); i++ {
		fmt.Println(lines[i])
	}

	return nil
}

// followLogs displays logs and follows new entries (like tail -f)
func followLogs(ctx context.Context, logFile string, tail int) error {
	// Set up signal handling for graceful shutdown
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	
	go func() {
		<-sigChan
		cancel()
	}()
	// First, show existing logs if tail is specified
	if tail > 0 {
		if err := showLogs(logFile, tail); err != nil {
			if os.IsNotExist(err) {
				fmt.Fprintf(os.Stderr, "Log file not found, waiting for it to be created: %s\n", logFile)
			} else {
				return err // Return other errors
			}
		}
	}

	var file *os.File
	var err error
	var lastPos int64

	// Helper to open and position the file
	openAndPositionFile := func() error {
		if file != nil {
			file.Close()
		}

		file, err = os.Open(logFile)
		if err != nil {
			return err
		}

		// If tail was used, we already showed the initial logs, so seek to end
		// Otherwise, show all content first, then follow
		if tail > 0 {
			lastPos, err = file.Seek(0, io.SeekEnd)
		} else {
			// If tail is 0, we want to show all existing content and then follow
			// This part is only relevant if the file exists from the start
			// If it doesn't exist, the initial showLogs would have handled it
			// or we'll just start following from creation.
			// So, if we reach here and tail is 0, it means we need to print all existing content
			// before starting to follow.
			currentSize, err := file.Seek(0, io.SeekEnd)
			if err != nil {
				return err
			}
			_, err = file.Seek(0, io.SeekStart)
			if err != nil {
				return err
			}
			if _, err := io.Copy(os.Stdout, file); err != nil {
				return err
			}
			lastPos = currentSize
		}
		return err
	}

	// Initial file opening
	err = openAndPositionFile()
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to open log file: %w", err)
	}

	// Main loop for following logs
	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}

		if file == nil {
			// File was not open initially or was rotated/deleted, try to open it
			err = openAndPositionFile()
			if err != nil {
				if os.IsNotExist(err) {
					// File still doesn't exist, wait and retry
					select {
					case <-ctx.Done():
						return nil
					case <-time.After(1 * time.Second):
					}
					continue
				}
				return fmt.Errorf("failed to open log file: %w", err)
			}
		}

		stat, err := os.Stat(logFile)
		if err != nil {
			if os.IsNotExist(err) {
				// File was rotated or deleted, close current file and set to nil to re-open
				if file != nil {
					file.Close()
					file = nil
				}
				select {
				case <-ctx.Done():
					return nil
				case <-time.After(100 * time.Millisecond):
				}
				continue
			}
			return fmt.Errorf("failed to stat log file: %w", err)
		}

		// Check if file was truncated (size is smaller than our position)
		if stat.Size() < lastPos {
			// File was truncated, seek to beginning
			lastPos, err = file.Seek(0, io.SeekStart)
			if err != nil {
				return fmt.Errorf("failed to seek to start: %w", err)
			}
		}

		reader := bufio.NewReader(file)
		// Seek to lastPos to ensure we read from where we left off
		_, err = file.Seek(lastPos, io.SeekStart)
		if err != nil {
			return fmt.Errorf("failed to seek to last position: %w", err)
		}

		line, err := reader.ReadString('\n')
		if err == io.EOF {
			// No new data, wait a bit
			select {
			case <-ctx.Done():
				return nil
			case <-time.After(100 * time.Millisecond):
			}
			continue
		}
		if err != nil {
			return fmt.Errorf("error reading log file: %w", err)
		}

		fmt.Print(line)
		lastPos += int64(len(line))
	}
}

// getAllLogFiles returns all log files including rotated ones
func getAllLogFiles(baseLogFile string) []string {
	var files []string
	files = append(files, baseLogFile)

	// Add rotated log files
	dir := filepath.Dir(baseLogFile)
	base := filepath.Base(baseLogFile)
	ext := filepath.Ext(base)
	name := strings.TrimSuffix(base, ext)

	for i := 1; i <= lifecycle.LogRotationCount; i++ {
		rotatedFile := filepath.Join(dir, name+"-"+strconv.Itoa(i)+ext)
		if _, err := os.Stat(rotatedFile); err == nil {
			files = append(files, rotatedFile)
		}
	}

	return files
}