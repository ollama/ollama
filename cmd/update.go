package cmd

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/updater"
	"github.com/ollama/ollama/version"
)

func NewUpdateCmd() *cobra.Command {
	updateCmd := &cobra.Command{
		Use:   "update",
		Short: "Check for or pull Ollama updates",
		RunE:  UpdateHandler,
	}

	updateCmd.Flags().Bool("check", false, "Check for updates without downloading")
	updateCmd.Flags().Bool("pull", false, "Pull the latest update")
	updateCmd.Flags().BoolP("force", "f", false, "Pull update even if already up to date")
	updateCmd.Flags().BoolP("install", "i", false, "Install the update after downloading")
	updateCmd.Flags().StringP("dir", "d", "", "Directory to download the update to")
	updateCmd.Flags().BoolP("yes", "y", false, "Automatically answer yes to prompts")
	updateCmd.Flags().String("url", "", "Custom URL for checking or downloading updates")
	updateCmd.Flags().Bool("rc", false, "Check for or pull release candidate (RC) versions")
	updateCmd.Flags().Bool("prerelease", false, "Alias for --rc")

	checkCmd := &cobra.Command{
		Use:   "check",
		Short: "Check for Ollama updates",
		Args:  cobra.NoArgs,
		RunE:  UpdateCheckHandler,
	}
	checkCmd.Flags().String("url", "", "Custom URL for checking updates")
	checkCmd.Flags().Bool("rc", false, "Check for release candidate (RC) versions")
	checkCmd.Flags().Bool("prerelease", false, "Alias for --rc")

	pullCmd := &cobra.Command{
		Use:   "pull",
		Short: "Pull the latest Ollama update",
		Args:  cobra.NoArgs,
		RunE:  UpdatePullHandler,
	}
	pullCmd.Flags().BoolP("force", "f", false, "Pull update even if already up to date")
	pullCmd.Flags().BoolP("install", "i", false, "Install the update after downloading")
	pullCmd.Flags().StringP("dir", "d", "", "Directory to download the update to")
	pullCmd.Flags().String("url", "", "Custom URL for downloading updates")
	pullCmd.Flags().BoolP("yes", "y", false, "Automatically answer yes to prompts")
	pullCmd.Flags().Bool("rc", false, "Pull release candidate (RC) versions")
	pullCmd.Flags().Bool("prerelease", false, "Alias for --rc")

	updateCmd.AddCommand(checkCmd, pullCmd)
	return updateCmd
}

func resolveCurrentVersion(cmd *cobra.Command) (currentVer string, serverVer string) {
	currentVer = version.Version
	if client, err := api.ClientFromEnvironment(); err == nil {
		if v, err := client.Version(cmd.Context()); err == nil && v != "" {
			serverVer = v
			if currentVer == "" || currentVer == "0.0.0" {
				currentVer = serverVer
			}
		}
	}
	return currentVer, serverVer
}

func UpdateHandler(cmd *cobra.Command, args []string) error {
	checkOnly, _ := cmd.Flags().GetBool("check")
	pullFlag, _ := cmd.Flags().GetBool("pull")

	if checkOnly && pullFlag {
		return errors.New("cannot specify both --check and --pull")
	}

	if checkOnly {
		return UpdateCheckHandler(cmd, args)
	}
	if pullFlag {
		return UpdatePullHandler(cmd, args)
	}

	urlFlag, _ := cmd.Flags().GetString("url")
	curVer, _ := resolveCurrentVersion(cmd)
	rcFlag, _ := cmd.Flags().GetBool("rc")
	prereleaseFlag, _ := cmd.Flags().GetBool("prerelease")
	info, err := updater.Check(cmd.Context(), updater.CheckOptions{
		CurrentVersion: curVer,
		URL:            urlFlag,
		IncludeRC:      rcFlag || prereleaseFlag,
	})
	if err != nil {
		return fmt.Errorf("failed to check for updates: %w", err)
	}

	displayUpdateStatus(cmd, info)

	if !info.UpdateAvailable {
		return nil
	}

	autoYes, _ := cmd.Flags().GetBool("yes")
	installFlag, _ := cmd.Flags().GetBool("install")
	if autoYes || installFlag {
		return doPull(cmd, info)
	}

	if isTerminalInteractive() {
		ok, err := launch.ConfirmPrompt(fmt.Sprintf("Download Ollama %s now?", info.LatestVersion))
		if err == nil && ok {
			return doPull(cmd, info)
		}
	}

	if info.IsRC {
		fmt.Println("\nRun 'ollama update pull --rc' to download this update.")
	} else {
		fmt.Println("\nRun 'ollama update pull' to download this update.")
	}
	return nil
}

func UpdateCheckHandler(cmd *cobra.Command, _ []string) error {
	urlFlag, _ := cmd.Flags().GetString("url")
	curVer, _ := resolveCurrentVersion(cmd)
	rcFlag, _ := cmd.Flags().GetBool("rc")
	prereleaseFlag, _ := cmd.Flags().GetBool("prerelease")
	info, err := updater.Check(cmd.Context(), updater.CheckOptions{
		CurrentVersion: curVer,
		URL:            urlFlag,
		IncludeRC:      rcFlag || prereleaseFlag,
	})
	if err != nil {
		return fmt.Errorf("failed to check for updates: %w", err)
	}

	displayUpdateStatus(cmd, info)

	if info.UpdateAvailable {
		if info.IsRC {
			fmt.Println("\nRun 'ollama update pull --rc' to download this update.")
		} else {
			fmt.Println("\nRun 'ollama update pull' to download this update.")
		}
	}
	return nil
}

func UpdatePullHandler(cmd *cobra.Command, _ []string) error {
	urlFlag, _ := cmd.Flags().GetString("url")
	curVer, _ := resolveCurrentVersion(cmd)
	rcFlag, _ := cmd.Flags().GetBool("rc")
	prereleaseFlag, _ := cmd.Flags().GetBool("prerelease")
	info, err := updater.Check(cmd.Context(), updater.CheckOptions{
		CurrentVersion: curVer,
		URL:            urlFlag,
		IncludeRC:      rcFlag || prereleaseFlag,
	})
	if err != nil {
		return fmt.Errorf("failed to check for updates: %w", err)
	}

	return doPull(cmd, info)
}

func doPull(cmd *cobra.Command, info *updater.ReleaseInfo) error {
	force, _ := cmd.Flags().GetBool("force")
	install, _ := cmd.Flags().GetBool("install")
	dir, _ := cmd.Flags().GetString("dir")

	if !force && !info.UpdateAvailable {
		fmt.Printf("Ollama is already up to date (%s).\nUse --force to download anyway.\n", info.CurrentVersion)
		return nil
	}

	asset := info.Asset
	if asset == nil {
		return fmt.Errorf("no download asset available for %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	fmt.Fprintf(os.Stderr, "pulling update %s (%s)...\n", info.LatestVersion, asset.Name)

	p := progress.NewProgress(os.Stderr)
	bar := progress.NewBar(asset.Name, asset.Size, 0)
	p.Add(asset.Name, bar)

	opts := updater.PullOptions{
		Dir:   dir,
		Force: force,
		ProgressFn: func(downloaded, total int64) {
			bar.Set(downloaded)
		},
	}

	res, err := updater.Pull(cmd.Context(), info, opts)
	p.Stop()
	if err != nil {
		return fmt.Errorf("failed to pull update: %w", err)
	}

	if res.AlreadyExisted {
		fmt.Printf("Update already downloaded: %s\n", res.FilePath)
	} else {
		fmt.Printf("Successfully pulled update: %s (%s)\n", res.FilePath, format.HumanBytes(res.Size))
	}

	if install {
		if err := doInstall(res.FilePath); err != nil {
			return err
		}
		for _, extra := range info.ExtraAssets {
			extraPath := filepath.Join(filepath.Dir(res.FilePath), extra.Name)
			if fi, err := os.Stat(extraPath); err == nil && fi.Size() > 0 {
				_ = doInstall(extraPath)
			}
		}
		return nil
	}

	printInstallInstructions(res.FilePath)
	return nil
}

func doInstall(archivePath string) error {
	installDir := updater.DefaultInstallDir()
	fmt.Printf("Installing update to %s...\n", installDir)
	if err := updater.Install(archivePath, installDir); err != nil {
		if os.IsPermission(err) {
			fmt.Fprintf(os.Stderr, "\nPermission denied. Please re-run with administrator privileges:\n  sudo ollama update --install\n")
			return err
		}
		return fmt.Errorf("installation failed: %w", err)
	}
	fmt.Println("Update installed successfully! Restart Ollama to apply changes (e.g. 'systemctl restart ollama').")
	return nil
}

func printInstallInstructions(archivePath string) {
	fmt.Println("\nTo install the update:")
	switch runtime.GOOS {
	case "linux":
		lower := strings.ToLower(archivePath)
		if strings.HasSuffix(lower, ".tar.zst") {
			fmt.Printf("  Run with sudo:\n    sudo ollama update --install\n  Or manually extract:\n    zstd -d < %s | sudo tar -C /usr/local -xf -\n", archivePath)
		} else {
			fmt.Printf("  Run with sudo:\n    sudo ollama update --install\n  Or manually extract:\n    sudo tar -C /usr/local -xzf %s\n", archivePath)
		}
	case "darwin":
		fmt.Printf("  Run:\n    ollama update --install\n  Or unzip %s and copy Ollama.app to /Applications\n", archivePath)
	case "windows":
		fmt.Printf("  Run installer:\n    %s\n", archivePath)
	default:
		fmt.Printf("  Extract %s to your installation directory.\n", archivePath)
	}
}

func displayUpdateStatus(cmd *cobra.Command, info *updater.ReleaseInfo) {
	serverVer := ""
	if client, err := api.ClientFromEnvironment(); err == nil {
		if v, err := client.Version(cmd.Context()); err == nil {
			serverVer = v
		}
	}

	if serverVer != "" && serverVer != info.CurrentVersion {
		fmt.Printf("Server version:   %s\n", serverVer)
		fmt.Printf("Client version:   %s\n", info.CurrentVersion)
	} else if info.IsDevVersion {
		fmt.Printf("Current version:  %s (development build)\n", info.CurrentVersion)
	} else if updater.IsRCVersion(info.CurrentVersion) {
		fmt.Printf("Current version:  %s (pre-release)\n", info.CurrentVersion)
	} else {
		fmt.Printf("Current version:  %s\n", info.CurrentVersion)
	}

	if info.IsRC {
		fmt.Printf("Latest version:   %s (pre-release)\n", info.LatestVersion)
	} else {
		fmt.Printf("Latest version:   %s\n", info.LatestVersion)
	}

	if info.UpdateAvailable {
		fmt.Println("Update available: yes")
	} else {
		fmt.Println("Update available: no (Ollama is up to date)")
	}

	if info.ReleaseURL != "" {
		fmt.Printf("Release URL:      %s\n", info.ReleaseURL)
	}
}

func isTerminalInteractive() bool {
	return term.IsTerminal(int(os.Stdin.Fd())) && term.IsTerminal(int(os.Stdout.Fd()))
}
