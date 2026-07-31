package updater

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	installScriptName              = "install.ps1"
	stagedCacheReadyFile           = "cache-ready"
	windowsCreateNoWindow          = 0x08000000
	installScriptModeCacheOnly     = "cache-only"
	installScriptModeInstallCached = "install-cached"
)

var (
	runningInstaller              string
	installScriptInstallerLogFile string
)

var (
	verifyInstallScriptSignature = verifyPowerShellScriptSignature
	runInstallScriptCacheOnly    = runInstallScriptCacheOnlyCommand
	startInstallScriptInstall    = startInstallScriptInstallCommand
	exitAfterStartingUpgrade     = os.Exit
)

type installScriptProcess interface {
	// Release lets the app detach from the launched install.ps1 process before
	// exiting; the interface keeps DoUpgradeAtStartup testable without starting
	// a real installer.
	Release() error
}

type installScriptCommand struct {
	Program       string
	Args          []string
	Env           []string
	Dir           string
	CreationFlags uint32
}

type OSVERSIONINFOEXW struct {
	dwOSVersionInfoSize uint32
	dwMajorVersion      uint32
	dwMinorVersion      uint32
	dwBuildNumber       uint32
	dwPlatformId        uint32
	szCSDVersion        [128]uint16
	wServicePackMajor   uint16
	wServicePackMinor   uint16
	wSuiteMask          uint16
	wProductType        uint8
	wReserved           uint8
}

func init() {
	VerifyDownload = verifyDownload
	Installer = "Ollama-darwin.zip"
	localAppData := os.Getenv("LOCALAPPDATA")
	appDataDir := filepath.Join(localAppData, "Ollama")

	// Use a distinct update staging directory from the old desktop app
	// to avoid double upgrades on the transition
	UpdateStageDir = filepath.Join(appDataDir, "updates_v2")

	UpgradeLogFile = filepath.Join(appDataDir, "upgrade.log")
	installScriptInstallerLogFile = filepath.Join(appDataDir, "OllamaSetup.log")
	Installer = "OllamaSetup.exe"
	runningInstaller = filepath.Join(appDataDir, Installer)
	UpgradeMarkerFile = filepath.Join(appDataDir, "upgraded")

	loadOSVersion()
}

func (u *Updater) downloadNewRelease(ctx context.Context, updateResp UpdateResponse) error {
	return u.downloadInstallScriptRelease(ctx, updateResp)
}

func (u *Updater) downloadInstallScriptRelease(ctx context.Context, updateResp UpdateResponse) error {
	scriptURL, err := installScriptURL(updateResp)
	if err != nil {
		return err
	}

	scriptPath, downloadedScript, err := downloadInstallScript(ctx, scriptURL, updateResp)
	if err != nil {
		return err
	}
	scriptStageDir := filepath.Dir(scriptPath)
	version := normalizedUpdateVersion(updateResp)

	if err := verifyInstallScriptSignature(scriptPath); err != nil {
		_ = os.RemoveAll(scriptStageDir)
		return fmt.Errorf("install.ps1 signature verification failed: %w", err)
	}
	if !installScriptSupportsCacheOnly(scriptPath) {
		_ = os.RemoveAll(scriptStageDir)
		return fmt.Errorf("install.ps1 does not support cache-only mode")
	}
	if !downloadedScript && isInstallScriptStageReady(scriptStageDir) {
		slog.Info("update already downloaded", "script", scriptPath)
		UpdateDownloaded = true
		return nil
	}

	// Clear readiness before refreshing the cache so a failed cache-only run
	// cannot leave startup upgrade stuck retrying stale or partial payloads.
	_ = removeInstallScriptStageReady(scriptStageDir)
	if err := runInstallScriptCacheOnly(ctx, scriptPath, scriptStageDir, version); err != nil {
		_ = os.RemoveAll(scriptStageDir)
		return err
	}

	if err := writeInstallScriptStageReady(scriptStageDir); err != nil {
		_ = os.RemoveAll(scriptStageDir)
		return err
	}
	UpdateDownloaded = true
	return nil
}

func downloadInstallScript(ctx context.Context, scriptURL string, updateResp UpdateResponse) (string, bool, error) {
	slog.Debug("checking install.ps1", "url", scriptURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, scriptURL, nil)
	if err != nil {
		return "", false, err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", false, fmt.Errorf("error checking install.ps1: %w", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", false, fmt.Errorf("unexpected status attempting to download install.ps1 %d", resp.StatusCode)
	}

	stageKey := strings.Join([]string{
		resp.Header.Get("etag"),
		normalizedUpdateVersion(updateResp),
		updateResp.UpdateURL,
	}, "\n")
	stageFilename, err := updateStagePath(UpdateStageDir, stageKey, installScriptName)
	if err != nil {
		return "", false, err
	}

	if _, err := os.Stat(stageFilename); err == nil {
		slog.Info("install.ps1 already downloaded", "script", stageFilename)
		return stageFilename, false, nil
	}

	cleanupOldDownloads(UpdateStageDir)

	slog.Debug("downloading install.ps1", "url", scriptURL, "script", stageFilename)
	req, err = http.NewRequestWithContext(ctx, http.MethodGet, scriptURL, nil)
	if err != nil {
		return "", false, err
	}
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		return "", false, fmt.Errorf("error downloading install.ps1: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", false, fmt.Errorf("unexpected status attempting to download install.ps1 %d", resp.StatusCode)
	}

	if err := os.MkdirAll(filepath.Dir(stageFilename), 0o755); err != nil {
		return "", false, fmt.Errorf("create update staging dir %s: %w", filepath.Dir(stageFilename), err)
	}

	tmpFilename := stageFilename + ".tmp"
	fp, err := os.OpenFile(tmpFilename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return "", false, fmt.Errorf("write install.ps1 %s: %w", tmpFilename, err)
	}
	if _, err := io.Copy(fp, resp.Body); err != nil {
		_ = fp.Close()
		_ = os.Remove(tmpFilename)
		return "", false, fmt.Errorf("write install.ps1 %s: %w", tmpFilename, err)
	}
	if err := fp.Close(); err != nil {
		_ = os.Remove(tmpFilename)
		return "", false, fmt.Errorf("close install.ps1 %s: %w", tmpFilename, err)
	}
	if err := os.Rename(tmpFilename, stageFilename); err != nil {
		_ = os.Remove(tmpFilename)
		return "", false, fmt.Errorf("stage install.ps1 %s: %w", stageFilename, err)
	}

	slog.Info("install.ps1 downloaded", "script", stageFilename)
	return stageFilename, true, nil
}

func installScriptURL(updateResp UpdateResponse) (string, error) {
	parsed, err := url.Parse(updateResp.UpdateURL)
	if err != nil {
		return "", fmt.Errorf("parse update URL: %w", err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("invalid update URL: %s", updateResp.UpdateURL)
	}
	parsed.Path = path.Join(path.Dir(parsed.Path), installScriptName)
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String(), nil
}

func normalizedUpdateVersion(updateResp UpdateResponse) string {
	return strings.TrimPrefix(strings.TrimSpace(updateResp.UpdateVersion), "v")
}

func windowsPowerShellPath() string {
	systemDir, err := windows.GetSystemDirectory()
	if err == nil && systemDir != "" {
		return filepath.Join(systemDir, "WindowsPowerShell", "v1.0", "powershell.exe")
	}
	slog.Warn("unable to resolve Windows system directory for powershell.exe", "error", err)

	systemRoot := os.Getenv("SystemRoot")
	if systemRoot == "" {
		systemRoot = os.Getenv("WINDIR")
	}
	if systemRoot == "" {
		systemRoot = `C:\Windows`
	}
	return filepath.Join(systemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
}

func validateWindowsPowerShellPath(powerShellPath string) error {
	info, err := os.Stat(powerShellPath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("Windows PowerShell not found at %s; app upgrades require the system Windows PowerShell component", powerShellPath)
		}
		return fmt.Errorf("unable to access Windows PowerShell at %s: %w", powerShellPath, err)
	}
	if info.IsDir() {
		return fmt.Errorf("Windows PowerShell path is a directory: %s", powerShellPath)
	}
	return nil
}

func windowsPowerShellModulePath() string {
	systemDir, err := windows.GetSystemDirectory()
	if err == nil && systemDir != "" {
		return filepath.Join(systemDir, "WindowsPowerShell", "v1.0", "Modules")
	}
	slog.Warn("unable to resolve Windows system directory for PowerShell module path", "error", err)

	systemRoot := os.Getenv("SystemRoot")
	if systemRoot == "" {
		systemRoot = os.Getenv("WINDIR")
	}
	if systemRoot == "" {
		systemRoot = `C:\Windows`
	}
	return filepath.Join(systemRoot, "System32", "WindowsPowerShell", "v1.0", "Modules")
}

func withWindowsPowerShellModulePath(base []string) []string {
	env := make([]string, 0, len(base)+1)
	for _, entry := range base {
		key, _, ok := strings.Cut(entry, "=")
		if ok && strings.EqualFold(key, "PSModulePath") {
			continue
		}
		env = append(env, entry)
	}
	// Force Windows PowerShell to its built-in module path. CI can inherit a
	// PowerShell 7 PSModulePath that cannot load Windows PowerShell modules, and
	// user-writable module paths should not participate in signature checks.
	return append(env, "PSModulePath="+windowsPowerShellModulePath())
}

func installScriptSupportsCacheOnly(scriptPath string) bool {
	data, err := os.ReadFile(scriptPath)
	if err != nil {
		slog.Warn("unable to read install.ps1 for capability check", "script", scriptPath, "error", err)
		return false
	}
	return strings.Contains(string(data), "OLLAMA_CACHE_ONLY")
}

func loadOSVersion() {
	UserAgentOS = "Windows"
	verInfo := OSVERSIONINFOEXW{}
	verInfo.dwOSVersionInfoSize = (uint32)(unsafe.Sizeof(verInfo))
	ntdll, err := windows.LoadDLL("ntdll.dll")
	if err != nil {
		slog.Warn("unable to find ntdll", "error", err)
		return
	}
	defer ntdll.Release()
	pRtlGetVersion, err := ntdll.FindProc("RtlGetVersion")
	if err != nil {
		slog.Warn("unable to locate RtlGetVersion", "error", err)
		return
	}
	status, _, err := pRtlGetVersion.Call(uintptr(unsafe.Pointer(&verInfo)))
	if status < 0x80000000 { // Success or Informational
		// Note: Windows 11 reports 10.0.22000 or newer
		UserAgentOS = fmt.Sprintf("Windows/%d.%d.%d", verInfo.dwMajorVersion, verInfo.dwMinorVersion, verInfo.dwBuildNumber)
	} else {
		slog.Warn("unable to get OS version", "error", err)
	}
}

func getStagedInstallScript() string {
	// When transitioning from old to new app, cleanup the update from the old staging dir
	// This can eventually be removed once enough time has passed since the transition
	cleanupOldDownloads(filepath.Join(os.Getenv("LOCALAPPDATA"), "Ollama", "updates"))

	files, err := filepath.Glob(filepath.Join(UpdateStageDir, "*", installScriptName))
	if err != nil {
		slog.Debug("failed to lookup downloads", "error", err)
		return ""
	}
	readyFiles := files[:0]
	for _, file := range files {
		if isInstallScriptStageReady(filepath.Dir(file)) {
			readyFiles = append(readyFiles, file)
		}
	}
	if len(readyFiles) == 0 {
		return ""
	} else if len(readyFiles) > 1 {
		// Shouldn't happen
		slog.Warn("multiple update downloads found, using first one", "bundles", readyFiles)
	}
	return readyFiles[0]
}

func DoUpgrade(interactive bool) error {
	if script := getStagedInstallScript(); script != "" {
		return doInstallScriptUpgrade(script, exitAfterStartingUpgrade)
	}
	return fmt.Errorf("failed to lookup downloads")
}

func doInstallScriptUpgrade(script string, exit func(int)) error {
	if script == "" {
		return fmt.Errorf("failed to lookup downloads")
	}

	if err := VerifyDownload(); err != nil {
		_ = os.RemoveAll(filepath.Dir(script))
		slog.Warn("verification failure", "script", script, "error", err)
		return fmt.Errorf("staged update verification failed: %w", err)
	}

	scriptStageDir := filepath.Dir(script)
	slog.Info("starting upgrade", "script", script, "stage", scriptStageDir)
	if err := createUpgradeMarker(); err != nil {
		slog.Warn("unable to create marker file", "file", UpgradeMarkerFile, "error", err)
	}

	if err := removeInstallScriptStageReady(scriptStageDir); err != nil {
		_ = os.Remove(UpgradeMarkerFile)
		return fmt.Errorf("unable to clear staged update readiness: %w", err)
	}

	command := newInstallScriptCommand(script, scriptStageDir, "", installScriptModeInstallCached)
	process, err := startInstallScriptInstall(command)
	if err != nil {
		_ = os.Remove(UpgradeMarkerFile)
		return fmt.Errorf("unable to start install.ps1 upgrade: %w", err)
	}
	if err := process.Release(); err != nil {
		slog.Error(fmt.Sprintf("failed to release installer process: %s", err))
	}

	slog.Info("install.ps1 upgrade started in background, exiting")
	exit(0)
	return nil
}

func DoPostUpgradeCleanup() error {
	if markerInfo, err := os.Stat(UpgradeMarkerFile); err == nil {
		logInstallerFailuresSince(markerInfo.ModTime())
	}

	cleanupOldDownloads(UpdateStageDir)
	err := os.Remove(UpgradeMarkerFile)
	if err != nil {
		slog.Warn("unable to clean up marker file", "marker", UpgradeMarkerFile, "error", err)
	}
	err = os.Remove(runningInstaller)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		slog.Debug("failed to remove running installer on first attempt, backgrounding...", "installer", runningInstaller, "error", err)
		go func() {
			for range 10 {
				time.Sleep(5 * time.Second)
				if err := os.Remove(runningInstaller); err == nil {
					slog.Debug("installer cleaned up")
					return
				}
				slog.Debug("failed to remove running installer on background attempt", "installer", runningInstaller, "error", err)
			}
		}()
	}
	return nil
}

func logInstallerFailuresSince(start time.Time) {
	for _, logFile := range []string{installScriptInstallerLogFile} {
		if logFile == "" {
			continue
		}
		failed, err := windowsInstallerLogIndicatesFailure(logFile, start)
		if err != nil {
			slog.Debug("unable to inspect installer log", "log", logFile, "error", err)
			continue
		}
		if failed {
			slog.Warn("Windows installer reported upgrade failure", "log", logFile)
		}
	}
}

func windowsInstallerLogIndicatesFailure(logFile string, since time.Time) (bool, error) {
	info, err := os.Stat(logFile)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	if info.ModTime().Before(since) {
		return false, nil
	}

	data, err := os.ReadFile(logFile)
	if err != nil {
		return false, err
	}
	logText := strings.ToLower(string(data))
	for _, marker := range []string{
		"installation process failed.",
		"installation process was aborted.",
		"user canceled the installation.",
	} {
		if strings.Contains(logText, marker) {
			return true, nil
		}
	}
	return false, nil
}

func verifyDownload() error {
	script := getStagedInstallScript()
	if script == "" {
		return fmt.Errorf("failed to lookup downloads")
	}
	slog.Debug("verifying update", "script", script)

	if err := verifyInstallScriptSignature(script); err != nil {
		return fmt.Errorf("signature verification failed: %w", err)
	}
	if !installScriptSupportsCacheOnly(script) {
		return fmt.Errorf("install.ps1 does not support cache-only mode")
	}
	return nil
}

func verifyPowerShellScriptSignature(filename string) error {
	verificationScript := powerShellSignatureVerificationScript(filename)
	powerShellPath := windowsPowerShellPath()
	if err := validateWindowsPowerShellPath(powerShellPath); err != nil {
		return err
	}
	cmd := exec.Command(
		powerShellPath,
		"-NoProfile",
		"-NonInteractive",
		"-Command",
		verificationScript,
	)
	cmd.Env = withWindowsPowerShellModulePath(os.Environ())
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: windowsCreateNoWindow}
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("Windows PowerShell signature verification failed using %s: %w: %s", powerShellPath, err, strings.TrimSpace(string(output)))
	}
	slog.Info("verified install.ps1 signature", "subject", strings.TrimSpace(string(output)))
	return nil
}

func powerShellSignatureVerificationScript(filename string) string {
	encodedFilename := base64.StdEncoding.EncodeToString([]byte(filename))
	return fmt.Sprintf(`
$ErrorActionPreference = 'Stop'
$target = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('%s'))
$sig = Get-AuthenticodeSignature -LiteralPath $target
if ($sig.Status -ne 'Valid') {
    throw "signature status: $($sig.Status)"
}
$subject = $sig.SignerCertificate.Subject
if ($subject -notmatch '(^|, )O=Ollama Inc\.(,|$)') {
    throw "unexpected signer: $subject"
}
Write-Output $subject
`, encodedFilename)
}

func runInstallScriptCacheOnlyCommand(ctx context.Context, scriptPath, scriptStageDir, version string) error {
	command := newInstallScriptCommand(scriptPath, scriptStageDir, version, installScriptModeCacheOnly)
	if err := validateWindowsPowerShellPath(command.Program); err != nil {
		return fmt.Errorf("install.ps1 cache phase failed: %w", err)
	}
	cmd := exec.CommandContext(ctx, command.Program, command.Args...)
	cmd.Env = command.Env
	cmd.Dir = command.Dir
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: command.CreationFlags}

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("install.ps1 cache phase failed using Windows PowerShell %s: %w: %s", command.Program, err, strings.TrimSpace(string(output)))
	}
	if len(output) > 0 {
		slog.Debug("install.ps1 cache phase completed", "output", strings.TrimSpace(string(output)))
	}
	return nil
}

func startInstallScriptInstallCommand(command installScriptCommand) (installScriptProcess, error) {
	if err := validateWindowsPowerShellPath(command.Program); err != nil {
		return nil, err
	}
	cmd := exec.Command(command.Program, command.Args...)
	cmd.Env = command.Env
	cmd.Dir = command.Dir
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: command.CreationFlags}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	if cmd.Process == nil {
		return nil, errors.New("install.ps1 process did not start")
	}
	return cmd.Process, nil
}

func newInstallScriptCommand(scriptPath, workingDir, version, mode string) installScriptCommand {
	env := installScriptEnv(os.Environ(), version, mode)
	return installScriptCommand{
		Program: windowsPowerShellPath(),
		Args: []string{
			"-NoProfile",
			// Execution policy is local machine/user policy, not our trust boundary.
			// Official builds verify install.ps1 before launch; updater_unsigned
			// test builds can disable only that verifier for local unsigned scripts.
			"-ExecutionPolicy",
			"Bypass",
			"-File",
			scriptPath,
		},
		Env:           env,
		Dir:           workingDir,
		CreationFlags: windowsCreateNoWindow,
	}
}

func installScriptEnv(base []string, version, mode string) []string {
	env := withWindowsPowerShellModulePath(filterInstallScriptEnv(base))
	switch mode {
	case installScriptModeCacheOnly:
		env = append(env, "OLLAMA_CACHE_ONLY=1")
		if version != "" {
			env = append(env, "OLLAMA_VERSION="+version)
		}
	case installScriptModeInstallCached:
		env = append(env, "OLLAMA_INSTALL_CACHED=1")
	}
	return env
}

func filterInstallScriptEnv(base []string) []string {
	blocked := map[string]struct{}{
		"OLLAMA_CACHE_ONLY":     {},
		"OLLAMA_DEBUG":          {},
		"OLLAMA_INSTALL_CACHED": {},
		"OLLAMA_INSTALL_DIR":    {},
		"OLLAMA_UNINSTALL":      {},
		"OLLAMA_VERSION":        {},
	}
	env := make([]string, 0, len(base))
	for _, entry := range base {
		key, _, ok := strings.Cut(entry, "=")
		if !ok {
			env = append(env, entry)
			continue
		}
		if _, found := blocked[strings.ToUpper(key)]; found {
			continue
		}
		env = append(env, entry)
	}
	return env
}

func createUpgradeMarker() error {
	if err := os.MkdirAll(filepath.Dir(UpgradeMarkerFile), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(UpgradeMarkerFile, os.O_RDONLY|os.O_CREATE, 0o666)
	if err != nil {
		return err
	}
	return f.Close()
}

func isInstallScriptStageReady(scriptStageDir string) bool {
	if _, err := os.Stat(filepath.Join(scriptStageDir, stagedCacheReadyFile)); err != nil {
		return false
	}
	return true
}

func writeInstallScriptStageReady(scriptStageDir string) error {
	if err := os.MkdirAll(scriptStageDir, 0o755); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(scriptStageDir, stagedCacheReadyFile), []byte("1\n"), 0o644)
}

func removeInstallScriptStageReady(scriptStageDir string) error {
	err := os.Remove(filepath.Join(scriptStageDir, stagedCacheReadyFile))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return err
}

func IsUpdatePending() bool {
	return getStagedInstallScript() != ""
}

func DoUpgradeAtStartup() error {
	return DoUpgrade(false)
}

func isInstallerRunning() bool {
	return len(IsProcRunning(Installer)) > 0
}

func IsProcRunning(procName string) []uint32 {
	pids := make([]uint32, 2048)
	var ret uint32
	if err := windows.EnumProcesses(pids, &ret); err != nil || ret == 0 {
		slog.Debug("failed to check for running installers", "error", err)
		return nil
	}
	pidCount := ret / uint32(unsafe.Sizeof(pids[0]))
	if pidCount > uint32(len(pids)) {
		pidCount = uint32(len(pids))
	}
	pids = pids[:pidCount]
	matches := []uint32{}
	for _, pid := range pids {
		if pid == 0 {
			continue
		}
		hProcess, err := windows.OpenProcess(windows.PROCESS_QUERY_INFORMATION|windows.PROCESS_VM_READ, false, pid)
		if err != nil {
			continue
		}
		defer windows.CloseHandle(hProcess)
		var module windows.Handle
		var cbNeeded uint32
		cb := (uint32)(unsafe.Sizeof(module))
		if err := windows.EnumProcessModules(hProcess, &module, cb, &cbNeeded); err != nil {
			continue
		}
		var sz uint32 = 1024 * 8
		moduleName := make([]uint16, sz)
		cb = uint32(len(moduleName)) * (uint32)(unsafe.Sizeof(uint16(0)))
		if err := windows.GetModuleBaseName(hProcess, module, &moduleName[0], cb); err != nil && err != syscall.ERROR_INSUFFICIENT_BUFFER {
			continue
		}
		exeFile := path.Base(strings.ToLower(syscall.UTF16ToString(moduleName)))
		if strings.EqualFold(exeFile, procName) {
			matches = append(matches, pid)
		}
	}
	return matches
}
