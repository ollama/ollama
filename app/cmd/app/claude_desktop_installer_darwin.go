//go:build darwin

package main

import (
	"archive/zip"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const (
	maxClaudeDesktopArchiveBytes = 1 << 30
	maxClaudeDesktopExtractBytes = 2 << 30
	maxClaudeDesktopArchiveFiles = 100_000
	claudeDesktopBundleID        = "com.anthropic.claudefordesktop"
	claudeDesktopTeamID          = "Q6L2SF6YDW"
)

var errClaudeDesktopDestinationExists = errors.New("Claude Desktop installation destination already exists")

func claudeDesktopInstallDestinations() []string {
	destinations := []string{"/Applications/Claude.app"}
	if home, err := os.UserHomeDir(); err == nil {
		destinations = append(destinations, filepath.Join(home, "Applications", "Claude.app"))
	}
	return destinations
}

func installClaudeDesktopZip(archivePath string, destinations []string, verify func(string) error) (string, error) {
	if len(destinations) == 0 {
		return "", errors.New("Claude Desktop installation destination is required")
	}
	if verify == nil {
		return "", errors.New("Claude Desktop bundle verifier is required")
	}
	info, err := os.Stat(archivePath)
	if err != nil {
		return "", fmt.Errorf("stat Claude Desktop archive: %w", err)
	}
	if !info.Mode().IsRegular() {
		return "", errors.New("Claude Desktop archive is not a regular file")
	}
	if info.Size() > maxClaudeDesktopArchiveBytes {
		return "", fmt.Errorf("Claude Desktop archive exceeds %d bytes", maxClaudeDesktopArchiveBytes)
	}

	workDir, err := os.MkdirTemp("", "ollama-claude-install-")
	if err != nil {
		return "", fmt.Errorf("create Claude Desktop installation directory: %w", err)
	}
	defer os.RemoveAll(workDir)

	if err := extractClaudeDesktopZip(archivePath, workDir); err != nil {
		return "", err
	}
	bundlePath := filepath.Join(workDir, "Claude.app")
	if err := validateClaudeDesktopBundle(bundlePath); err != nil {
		return "", err
	}
	if err := verify(bundlePath); err != nil {
		return "", fmt.Errorf("verify Claude Desktop signature: %w", err)
	}

	var permissionErr error
	for _, destination := range destinations {
		if strings.TrimSpace(destination) == "" {
			continue
		}
		if _, err := os.Stat(destination); err == nil {
			return "", fmt.Errorf("%w: %s", errClaudeDesktopDestinationExists, destination)
		} else if !errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("check Claude Desktop destination %s: %w", destination, err)
		}
		if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
			if errors.Is(err, os.ErrPermission) {
				permissionErr = err
				continue
			}
			return "", fmt.Errorf("create Claude Desktop destination: %w", err)
		}
		if err := os.Rename(bundlePath, destination); err != nil {
			if errors.Is(err, os.ErrPermission) {
				permissionErr = err
				continue
			}
			return "", fmt.Errorf("move Claude Desktop to %s: %w", destination, err)
		}
		return destination, nil
	}
	if permissionErr != nil {
		return "", fmt.Errorf("install Claude Desktop in Applications: %w", permissionErr)
	}
	return "", errors.New("Claude Desktop installation destination is required")
}

func extractClaudeDesktopZip(archivePath, destination string) error {
	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("open Claude Desktop archive: %w", err)
	}
	defer reader.Close()
	if len(reader.File) == 0 {
		return errors.New("Claude Desktop archive is empty")
	}
	if len(reader.File) > maxClaudeDesktopArchiveFiles {
		return fmt.Errorf("Claude Desktop archive contains more than %d files", maxClaudeDesktopArchiveFiles)
	}

	var expanded uint64
	for _, file := range reader.File {
		clean, err := safeClaudeDesktopArchivePath(file.Name)
		if err != nil {
			return err
		}
		expanded += file.UncompressedSize64
		if expanded > maxClaudeDesktopExtractBytes {
			return fmt.Errorf("Claude Desktop archive expands beyond %d bytes", maxClaudeDesktopExtractBytes)
		}
		path := filepath.Join(destination, filepath.FromSlash(clean))
		switch {
		case file.FileInfo().IsDir():
			if err := os.MkdirAll(path, file.Mode().Perm()); err != nil {
				return fmt.Errorf("create Claude Desktop archive directory: %w", err)
			}
		case file.Mode()&os.ModeSymlink != 0:
			target, err := readClaudeDesktopZipFile(file, 16<<10)
			if err != nil {
				return fmt.Errorf("read Claude Desktop archive symlink: %w", err)
			}
			if err := validateClaudeDesktopSymlink(clean, string(target)); err != nil {
				return err
			}
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				return fmt.Errorf("create Claude Desktop archive directory: %w", err)
			}
			if err := os.Symlink(string(target), path); err != nil {
				return fmt.Errorf("create Claude Desktop archive symlink: %w", err)
			}
		case file.Mode().IsRegular():
			if err := extractClaudeDesktopZipFile(file, path); err != nil {
				return err
			}
		default:
			return fmt.Errorf("Claude Desktop archive contains unsupported file %q", file.Name)
		}
	}
	return nil
}

func safeClaudeDesktopArchivePath(name string) (string, error) {
	if strings.ContainsRune(name, '\x00') || filepath.IsAbs(name) {
		return "", fmt.Errorf("Claude Desktop archive contains unsafe path %q", name)
	}
	clean := filepath.ToSlash(filepath.Clean(name))
	if clean != "Claude.app" && !strings.HasPrefix(clean, "Claude.app/") {
		return "", fmt.Errorf("Claude Desktop archive contains unexpected path %q", name)
	}
	return clean, nil
}

func validateClaudeDesktopSymlink(name, target string) error {
	if target == "" || filepath.IsAbs(target) {
		return fmt.Errorf("Claude Desktop archive contains unsafe symlink %q", name)
	}
	resolved := filepath.Clean(filepath.Join(filepath.Dir(name), target))
	resolved = filepath.ToSlash(resolved)
	if resolved != "Claude.app" && !strings.HasPrefix(resolved, "Claude.app/") {
		return fmt.Errorf("Claude Desktop archive symlink %q escapes Claude.app", name)
	}
	return nil
}

func extractClaudeDesktopZipFile(file *zip.File, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create Claude Desktop archive directory: %w", err)
	}
	input, err := file.Open()
	if err != nil {
		return fmt.Errorf("open Claude Desktop archive file: %w", err)
	}
	output, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, file.Mode().Perm())
	if err != nil {
		input.Close()
		return fmt.Errorf("create Claude Desktop archive file: %w", err)
	}
	_, copyErr := io.Copy(output, input)
	inputErr := input.Close()
	outputErr := output.Close()
	if copyErr != nil {
		return fmt.Errorf("extract Claude Desktop archive file: %w", copyErr)
	}
	if inputErr != nil {
		return fmt.Errorf("close Claude Desktop archive file: %w", inputErr)
	}
	if outputErr != nil {
		return fmt.Errorf("close extracted Claude Desktop file: %w", outputErr)
	}
	return nil
}

func readClaudeDesktopZipFile(file *zip.File, limit int64) ([]byte, error) {
	reader, err := file.Open()
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	data, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("archive entry exceeds %d bytes", limit)
	}
	return data, nil
}

func validateClaudeDesktopBundle(bundlePath string) error {
	info, err := os.Stat(bundlePath)
	if err != nil || !info.IsDir() {
		return errors.New("Claude Desktop archive does not contain Claude.app")
	}
	executable := filepath.Join(bundlePath, "Contents", "MacOS", "Claude")
	info, err = os.Stat(executable)
	if err != nil {
		return fmt.Errorf("Claude Desktop executable is missing: %w", err)
	}
	if !info.Mode().IsRegular() || info.Mode()&0o111 == 0 {
		return errors.New("Claude Desktop executable is not executable")
	}
	return nil
}

func verifyClaudeDesktopBundle(bundlePath string) error {
	if output, err := exec.Command("/usr/bin/codesign", "--verify", "--deep", "--strict", bundlePath).CombinedOutput(); err != nil {
		return fmt.Errorf("codesign verification failed: %w: %s", err, strings.TrimSpace(string(output)))
	}
	output, err := exec.Command("/usr/bin/codesign", "-d", "--verbose=4", bundlePath).CombinedOutput()
	if err != nil {
		return fmt.Errorf("read code signature: %w: %s", err, strings.TrimSpace(string(output)))
	}
	details := string(output)
	if !strings.Contains(details, "Identifier="+claudeDesktopBundleID) ||
		!strings.Contains(details, "TeamIdentifier="+claudeDesktopTeamID) {
		return fmt.Errorf("unexpected Claude Desktop signing identity")
	}
	return nil
}
