package updater

import (
	"errors"
	"fmt"
	"log/slog"
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

var runningInstaller string

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
	Installer = "OllamaSetup.exe"
	runningInstaller = filepath.Join(appDataDir, Installer)
	UpgradeMarkerFile = filepath.Join(appDataDir, "upgraded")

	loadOSVersion()
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

func getStagedUpdate() string {
	// When transitioning from old to new app, cleanup the update from the old staging dir
	// This can eventually be removed once enough time has passed since the transition
	cleanupOldDownloads(filepath.Join(os.Getenv("LOCALAPPDATA"), "Ollama", "updates"))

	files, err := filepath.Glob(filepath.Join(UpdateStageDir, "*", "*.exe"))
	if err != nil {
		slog.Debug("failed to lookup downloads", "error", err)
		return ""
	}
	if len(files) == 0 {
		return ""
	} else if len(files) > 1 {
		// Shouldn't happen
		slog.Warn("multiple update downloads found, using first one", "bundles", files)
	}
	return files[0]
}

func DoUpgrade(interactive bool) error {
	bundle := getStagedUpdate()
	if bundle == "" {
		return fmt.Errorf("failed to lookup downloads")
	}

	// We move the installer to ensure we don't race with multiple apps starting in quick succession
	if err := os.Rename(bundle, runningInstaller); err != nil {
		return fmt.Errorf("unable to rename %s -> %s : %w", bundle, runningInstaller, err)
	}

	slog.Info("upgrade log file " + UpgradeLogFile)

	// make the upgrade show progress, but non interactive
	installArgs := []string{
		"/CLOSEAPPLICATIONS",                    // Quit the tray app if it's still running
		"/LOG=" + filepath.Base(UpgradeLogFile), // Only relative seems reliable, so set pwd
		"/FORCECLOSEAPPLICATIONS",               // Force close the tray app - might be needed
		"/SP",                                   // Skip the "This will install... Do you wish to continue" prompt
		"/NOCANCEL",                             // Disable the ability to cancel upgrade mid-flight to avoid partially installed upgrades
		"/SILENT",
	}

	if !interactive {
		// Add flags to make it totally silent without GUI
		installArgs = append(installArgs, "/VERYSILENT", "/SUPPRESSMSGBOXES")
	}

	slog.Info("starting upgrade", "installer", runningInstaller, "args", installArgs)
	os.Chdir(filepath.Dir(UpgradeLogFile)) //nolint:errcheck
	cmd := exec.Command(runningInstaller, installArgs...)

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("unable to start ollama app %w", err)
	}

	if cmd.Process != nil {
		err := cmd.Process.Release()
		if err != nil {
			slog.Error(fmt.Sprintf("failed to release server process: %s", err))
		}
	} else {
		// TODO - some details about why it didn't start, or is this a pedantic error case?
		return errors.New("installer process did not start")
	}

	// If the install fails to upgrade the system, and leaves a functional
	// app, this marker file will cause us to remove the staged upgrade
	// bundle, which will prevent trying again until we download again.
	// If this becomes looping a problem, we may need to look for failures
	// in the upgrade log in DoPostUpgradeCleanup and then not download
	// the same version again.
	f, err := os.OpenFile(UpgradeMarkerFile, os.O_RDONLY|os.O_CREATE, 0o666)
	if err != nil {
		slog.Warn("unable to create marker file", "file", UpgradeMarkerFile, "error", err)
	}
	f.Close()

	// TODO should we linger for a moment and check to make sure it's actually running by checking the pid?

	slog.Info("Installer started in background, exiting")

	os.Exit(0)
	// Not reached
	return nil
}

func DoPostUpgradeCleanup() error {
	cleanupOldDownloads(UpdateStageDir)
	err := os.Remove(UpgradeMarkerFile)
	if err != nil {
		slog.Warn("unable to clean up marker file", "marker", UpgradeMarkerFile, "error", err)
	}
	err = os.Remove(runningInstaller)
	if err != nil {
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

func verifyDownload() error {
	return nil
}

func IsUpdatePending() bool {
	return getStagedUpdate() != ""
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
	pids = pids[:ret]
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
