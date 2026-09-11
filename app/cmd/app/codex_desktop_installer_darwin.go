//go:build darwin

package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const (
	maxCodexDesktopDiskImageBytes = 2 << 30
	codexDesktopBundleID          = "com.openai.codex"
	codexDesktopTeamID            = "2DC432GLL2"
)

var errCodexDesktopDestinationExists = errors.New("ChatGPT installation destination already exists")

func codexDesktopInstallDestinations() []string {
	destinations := []string{"/Applications/ChatGPT.app"}
	if home, err := os.UserHomeDir(); err == nil {
		destinations = append(destinations, filepath.Join(home, "Applications", "ChatGPT.app"))
	}
	return destinations
}

func installCodexDesktopDiskImage(imagePath string, destinations []string, verify func(string) error) (installedPath string, err error) {
	if len(destinations) == 0 {
		return "", errors.New("ChatGPT installation destination is required")
	}
	if verify == nil {
		return "", errors.New("ChatGPT bundle verifier is required")
	}
	info, err := os.Stat(imagePath)
	if err != nil {
		return "", fmt.Errorf("stat ChatGPT disk image: %w", err)
	}
	if !info.Mode().IsRegular() {
		return "", errors.New("ChatGPT disk image is not a regular file")
	}
	if info.Size() > maxCodexDesktopDiskImageBytes {
		return "", fmt.Errorf("ChatGPT disk image exceeds %d bytes", maxCodexDesktopDiskImageBytes)
	}

	workDir, err := os.MkdirTemp("", "ollama-chatgpt-install-")
	if err != nil {
		return "", fmt.Errorf("create ChatGPT installation directory: %w", err)
	}
	defer os.RemoveAll(workDir)
	mountPath := filepath.Join(workDir, "volume")
	if err := os.Mkdir(mountPath, 0o700); err != nil {
		return "", fmt.Errorf("create ChatGPT mount point: %w", err)
	}

	output, err := exec.Command(
		"/usr/bin/hdiutil",
		"attach",
		"-nobrowse",
		"-readonly",
		"-mountpoint",
		mountPath,
		imagePath,
	).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("mount ChatGPT disk image: %w: %s", err, strings.TrimSpace(string(output)))
	}
	defer func() {
		detachOutput, detachErr := exec.Command("/usr/bin/hdiutil", "detach", mountPath).CombinedOutput()
		if detachErr == nil {
			return
		}
		forceOutput, forceErr := exec.Command("/usr/bin/hdiutil", "detach", "-force", mountPath).CombinedOutput()
		if forceErr != nil && err == nil {
			err = fmt.Errorf(
				"unmount ChatGPT disk image: %v: %s; force detach: %v: %s",
				detachErr,
				strings.TrimSpace(string(detachOutput)),
				forceErr,
				strings.TrimSpace(string(forceOutput)),
			)
		}
	}()

	bundlePath, err := codexDesktopBundleOnVolume(mountPath)
	if err != nil {
		return "", err
	}
	return installCodexDesktopBundle(bundlePath, destinations, verify)
}

func codexDesktopBundleOnVolume(mountPath string) (string, error) {
	for _, name := range []string{"ChatGPT.app", "Codex.app"} {
		bundlePath := filepath.Join(mountPath, name)
		info, err := os.Lstat(bundlePath)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return "", fmt.Errorf("inspect ChatGPT bundle: %w", err)
		}
		if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
			return "", fmt.Errorf("ChatGPT disk image contains an invalid %s", name)
		}
		return bundlePath, nil
	}
	return "", errors.New("ChatGPT disk image does not contain ChatGPT.app")
}

func installCodexDesktopBundle(bundlePath string, destinations []string, verify func(string) error) (string, error) {
	if err := validateCodexDesktopBundle(bundlePath); err != nil {
		return "", err
	}
	if err := verify(bundlePath); err != nil {
		return "", fmt.Errorf("verify ChatGPT signature: %w", err)
	}

	var permissionErr error
	for _, destination := range destinations {
		if strings.TrimSpace(destination) == "" {
			continue
		}
		if _, err := os.Lstat(destination); err == nil {
			return "", fmt.Errorf("%w: %s", errCodexDesktopDestinationExists, destination)
		} else if !errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("check ChatGPT destination %s: %w", destination, err)
		}
		parent := filepath.Dir(destination)
		if err := os.MkdirAll(parent, 0o755); err != nil {
			if errors.Is(err, os.ErrPermission) {
				permissionErr = err
				continue
			}
			return "", fmt.Errorf("create ChatGPT destination: %w", err)
		}
		stageDir, err := os.MkdirTemp(parent, ".ollama-chatgpt-install-")
		if err != nil {
			if errors.Is(err, os.ErrPermission) {
				permissionErr = err
				continue
			}
			return "", fmt.Errorf("create staged ChatGPT destination: %w", err)
		}
		stagedBundle := filepath.Join(stageDir, "ChatGPT.app")
		copyOutput, copyErr := exec.Command("/usr/bin/ditto", bundlePath, stagedBundle).CombinedOutput()
		if copyErr == nil {
			copyErr = validateCodexDesktopBundle(stagedBundle)
		}
		if copyErr == nil {
			copyErr = verify(stagedBundle)
		}
		if copyErr == nil {
			copyErr = os.Rename(stagedBundle, destination)
		}
		removeErr := os.RemoveAll(stageDir)
		if copyErr != nil {
			if errors.Is(copyErr, os.ErrPermission) {
				permissionErr = copyErr
				continue
			}
			return "", fmt.Errorf("install ChatGPT in %s: %w: %s", parent, copyErr, strings.TrimSpace(string(copyOutput)))
		}
		if removeErr != nil {
			return "", fmt.Errorf("remove staged ChatGPT destination: %w", removeErr)
		}
		return destination, nil
	}
	if permissionErr != nil {
		return "", fmt.Errorf("install ChatGPT in Applications: %w", permissionErr)
	}
	return "", errors.New("ChatGPT installation destination is required")
}

func validateCodexDesktopBundle(bundlePath string) error {
	info, err := os.Stat(bundlePath)
	if err != nil || !info.IsDir() {
		return errors.New("ChatGPT disk image does not contain a valid app bundle")
	}
	for _, executableName := range []string{"ChatGPT", "Codex"} {
		executable := filepath.Join(bundlePath, "Contents", "MacOS", executableName)
		info, err = os.Stat(executable)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return fmt.Errorf("inspect ChatGPT executable: %w", err)
		}
		if info.Mode().IsRegular() && info.Mode()&0o111 != 0 {
			return nil
		}
		return errors.New("ChatGPT executable is not executable")
	}
	return errors.New("ChatGPT executable is missing")
}

func verifyCodexDesktopBundle(bundlePath string) error {
	if output, err := exec.Command("/usr/bin/codesign", "--verify", "--deep", "--strict", bundlePath).CombinedOutput(); err != nil {
		return fmt.Errorf("codesign verification failed: %w: %s", err, strings.TrimSpace(string(output)))
	}
	output, err := exec.Command("/usr/bin/codesign", "-d", "--verbose=4", bundlePath).CombinedOutput()
	if err != nil {
		return fmt.Errorf("read code signature: %w: %s", err, strings.TrimSpace(string(output)))
	}
	details := string(output)
	if !strings.Contains(details, "Identifier="+codexDesktopBundleID) ||
		!strings.Contains(details, "TeamIdentifier="+codexDesktopTeamID) {
		return errors.New("unexpected ChatGPT signing identity")
	}
	return nil
}
