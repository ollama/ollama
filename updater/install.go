package updater

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/klauspost/compress/zstd"
)

// DefaultInstallDir returns the default installation directory for the current platform.
func DefaultInstallDir() string {
	switch runtime.GOOS {
	case "darwin":
		return "/Applications"
	case "windows":
		if localAppData := os.Getenv("LOCALAPPDATA"); localAppData != "" {
			return filepath.Join(localAppData, "Programs", "Ollama")
		}
		return `C:\Program Files\Ollama`
	default:
		// Linux/other
		exe, err := os.Executable()
		if err == nil {
			exeDir := filepath.Dir(exe)
			if strings.HasSuffix(exeDir, "/bin") {
				return filepath.Dir(exeDir) // e.g. /usr/local
			}
		}
		if _, err := os.Stat("/usr/local/bin"); err == nil {
			return "/usr/local"
		}
		return "/usr"
	}
}

// Install extracts the update archive into the specified destination directory.
func Install(archivePath string, installDir string) error {
	if installDir == "" {
		installDir = DefaultInstallDir()
	}

	installDir, err := filepath.Abs(installDir)
	if err != nil {
		return fmt.Errorf("resolve install dir: %w", err)
	}

	if err := os.MkdirAll(installDir, 0o755); err != nil {
		return fmt.Errorf("create install directory %s: %w", installDir, err)
	}

	lower := strings.ToLower(archivePath)
	switch {
	case strings.HasSuffix(lower, ".tar.zst"):
		return extractTarZst(archivePath, installDir)
	case strings.HasSuffix(lower, ".tar.gz") || strings.HasSuffix(lower, ".tgz"):
		return extractTarGz(archivePath, installDir)
	case strings.HasSuffix(lower, ".zip"):
		return extractZip(archivePath, installDir)
	default:
		return fmt.Errorf("unsupported update archive format: %s", filepath.Base(archivePath))
	}
}

func extractTarZst(archivePath, destDir string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return fmt.Errorf("open archive: %w", err)
	}
	defer f.Close()

	zr, err := zstd.NewReader(f)
	if err != nil {
		return fmt.Errorf("create zstd reader: %w", err)
	}
	defer zr.Close()

	return extractTarReader(tar.NewReader(zr), destDir)
}

func extractTarGz(archivePath, destDir string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return fmt.Errorf("open archive: %w", err)
	}
	defer f.Close()

	gr, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("create gzip reader: %w", err)
	}
	defer gr.Close()

	return extractTarReader(tar.NewReader(gr), destDir)
}

func extractTarReader(tr *tar.Reader, destDir string) error {
	for {
		header, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return fmt.Errorf("read tar entry: %w", err)
		}

		targetPath, err := safeDestinationPath(destDir, header.Name)
		if err != nil {
			return err
		}
		targetPath = resolveBinaryTarget(targetPath)

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(targetPath, 0o755); err != nil {
				return fmt.Errorf("create dir %s: %w", targetPath, err)
			}
		case tar.TypeReg, tar.TypeRegA:
			if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
				return fmt.Errorf("create parent dir for %s: %w", targetPath, err)
			}
			mode := header.FileInfo().Mode().Perm()
			if mode == 0 {
				mode = 0o755
			}
			// Write to a temporary file in the same directory and atomically rename.
			// This avoids ETXTBSY if the target executable is currently running,
			// and ensures atomic replacement so running daemons are not corrupted.
			tmpPath := targetPath + ".new"
			out, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
			if err != nil {
				return fmt.Errorf("create file %s: %w", tmpPath, err)
			}
			if _, err := io.Copy(out, tr); err != nil {
				out.Close()
				_ = os.Remove(tmpPath)
				return fmt.Errorf("write file %s: %w", tmpPath, err)
			}
			if err := out.Close(); err != nil {
				_ = os.Remove(tmpPath)
				return fmt.Errorf("close file %s: %w", tmpPath, err)
			}
			if err := os.Rename(tmpPath, targetPath); err != nil {
				_ = os.Remove(tmpPath)
				return fmt.Errorf("replace file %s: %w", targetPath, err)
			}
		case tar.TypeSymlink:
			if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
				return fmt.Errorf("create parent dir for symlink %s: %w", targetPath, err)
			}
			_ = os.Remove(targetPath)
			if err := os.Symlink(header.Linkname, targetPath); err != nil {
				return fmt.Errorf("create symlink %s -> %s: %w", targetPath, header.Linkname, err)
			}
		}
	}
	return nil
}

func extractZip(archivePath, destDir string) error {
	r, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("open zip: %w", err)
	}
	defer r.Close()

	for _, f := range r.File {
		targetPath, err := safeDestinationPath(destDir, f.Name)
		if err != nil {
			return err
		}
		targetPath = resolveBinaryTarget(targetPath)

		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(targetPath, 0o755); err != nil {
				return fmt.Errorf("create dir %s: %w", targetPath, err)
			}
			continue
		}

		if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
			return fmt.Errorf("create parent dir for %s: %w", targetPath, err)
		}

		rc, err := f.Open()
		if err != nil {
			return fmt.Errorf("open zip entry %s: %w", f.Name, err)
		}

		mode := f.Mode().Perm()
		if mode == 0 {
			mode = 0o755
		}
		tmpPath := targetPath + ".new"
		out, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
		if err != nil {
			rc.Close()
			return fmt.Errorf("create file %s: %w", tmpPath, err)
		}

		_, copyErr := io.Copy(out, rc)
		rc.Close()
		closeErr := out.Close()
		if copyErr != nil {
			_ = os.Remove(tmpPath)
			return fmt.Errorf("extract zip entry %s: %w", f.Name, copyErr)
		}
		if closeErr != nil {
			_ = os.Remove(tmpPath)
			return fmt.Errorf("close file %s: %w", tmpPath, closeErr)
		}
		if err := os.Rename(tmpPath, targetPath); err != nil {
			_ = os.Remove(tmpPath)
			return fmt.Errorf("replace file %s: %w", targetPath, err)
		}
	}
	return nil
}

func safeDestinationPath(destDir, entryName string) (string, error) {
	cleanName := filepath.Clean(filepath.FromSlash(entryName))
	if filepath.IsAbs(cleanName) || cleanName == ".." || strings.HasPrefix(cleanName, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("unsafe entry path %q in archive", entryName)
	}

	fullPath := filepath.Join(destDir, cleanName)
	rel, err := filepath.Rel(destDir, fullPath)
	if err != nil || strings.HasPrefix(rel, "..") || rel == "." && cleanName != "." {
		return "", fmt.Errorf("entry %q escapes destination directory %s", entryName, destDir)
	}

	return fullPath, nil
}

// resolveBinaryTarget checks whether targetPath (e.g. /usr/local/bin/ollama) is a wrapper script
// pointing to a real binary (e.g. ollama.bin or specified by REAL_BIN). If so, it returns
// the path to the real binary so the launcher wrapper script is preserved while updating the ELF binary.
func resolveBinaryTarget(targetPath string) string {
	base := filepath.Base(targetPath)
	if base != "ollama" && base != "ollama.exe" {
		return targetPath
	}

	fi, err := os.Lstat(targetPath)
	if err != nil {
		return targetPath
	}

	// If it's a symlink, resolve to the underlying target
	if fi.Mode()&os.ModeSymlink != 0 {
		resolved, err := filepath.EvalSymlinks(targetPath)
		if err == nil && resolved != targetPath {
			return resolved
		}
		return targetPath
	}

	// Check if it's a shell script wrapper
	f, err := os.Open(targetPath)
	if err != nil {
		return targetPath
	}
	defer f.Close()

	buf := make([]byte, 2048)
	n, _ := f.Read(buf)
	if n > 2 && string(buf[:2]) == "#!" {
		content := string(buf[:n])
		dir := filepath.Dir(targetPath)

		// Look for REAL_BIN or ollama.bin in the wrapper script
		if strings.Contains(content, "ollama.bin") || strings.Contains(content, "REAL_BIN") {
			if envReal := os.Getenv("OLLAMA_REAL_BIN"); envReal != "" {
				return envReal
			}
			return filepath.Join(dir, "ollama.bin")
		}
	}

	return targetPath
}
