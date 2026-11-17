package mlxrunner

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"debug/macho"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
)

// executableBuildID reads Go's embedded build stamp, without requiring a Go
// installation or hashing the executable. Missing IDs disable persistent reuse.
func executableBuildID(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	data := make([]byte, 64<<10)
	n, err := f.ReadAt(data, 0)
	if err != nil && err != io.EOF {
		return "", err
	}
	data = data[:n]
	if len(data) >= 4 && binary.BigEndian.Uint32(data) == macho.MagicFat {
		fat, err := macho.NewFatFile(f)
		if err != nil {
			return "", err
		}
		cpu := macho.CpuArm64
		if runtime.GOARCH == "amd64" {
			cpu = macho.CpuAmd64
		}
		found := false
		for _, arch := range fat.Arches {
			if arch.Cpu == cpu {
				n, err = f.ReadAt(data, int64(arch.Offset))
				if err != nil && err != io.EOF {
					return "", err
				}
				data = data[:n]
				found = true
				break
			}
		}
		if !found {
			return "", fmt.Errorf("universal executable has no %s slice", runtime.GOARCH)
		}
	}
	if bytes.HasPrefix(data, []byte("\x7fELF")) {
		obj, err := elf.NewFile(f)
		if err != nil {
			return "", err
		}
		section := obj.Section(".note.go.buildid")
		if section != nil && section.Size <= 4096 {
			note, err := section.Data()
			if err != nil {
				return "", err
			}
			if len(note) >= 16 && obj.ByteOrder.Uint32(note[:4]) == 4 && obj.ByteOrder.Uint32(note[8:12]) == 4 && string(note[12:16]) == "Go\x00\x00" {
				size := obj.ByteOrder.Uint32(note[4:8])
				if size > 0 && uint64(size) <= uint64(len(note)-16) {
					return string(note[16 : 16+size]), nil
				}
			}
		}
	} else {
		// Go's Mach-O and PE linkers emit this quoted marker near the header.
		const marker = "\xff Go build ID: "
		if i := bytes.Index(data, []byte(marker)); i >= 0 {
			tail := data[i+len(marker):]
			if end := bytes.Index(tail, []byte("\n \xff")); end > 0 {
				id, err := strconv.Unquote(string(tail[:end]))
				if err == nil && id != "" {
					return id, nil
				}
			}
		}
	}
	return "", fmt.Errorf("executable has no readable Go build ID")
}

// A runner process loads one native payload. Its content identity is shared
// by the generation and readout cache layouts.
var nativeCacheIdentity = sync.OnceValues(func() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	id, err := executableBuildID(exe)
	if err != nil {
		return "", err
	}
	library, err := mlx.LoadedLibraryPath()
	if err != nil {
		return "", err
	}
	// The dynamically loaded payload is independent of the Go link. Include the
	// adjacent MLX library and kernel resources, including symlink targets.
	entries, err := os.ReadDir(filepath.Dir(library))
	if err != nil {
		return "", err
	}
	h := sha256.New()
	fmt.Fprintln(h, "ollama-kv-v1", id)
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !(strings.HasPrefix(name, "libmlx") || strings.HasPrefix(name, "mlx") || strings.HasSuffix(name, ".metallib")) {
			continue
		}
		f, err := os.Open(filepath.Join(filepath.Dir(library), name))
		if err != nil {
			return "", err
		}
		fmt.Fprintln(h, name)
		_, err = io.Copy(h, f)
		closeErr := f.Close()
		if err != nil {
			return "", err
		}
		if closeErr != nil {
			return "", closeErr
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
})

// cacheLayout is an opaque partition identity supplied before model loading.
// Split setup must include the rank, world size, split kind and full partition
// assignment. An empty tag keeps the single-runner identity unchanged.
type cacheLayout string

func (l cacheLayout) writeTo(w io.Writer) {
	if l != "" {
		fmt.Fprintf(w, "layout %q\n", l)
	}
}

func cacheIdentity(m *manifest.Manifest, c *prefixCache, layout cacheLayout) (string, string, error) {
	runtimeID, err := nativeCacheIdentity()
	if err != nil {
		return "", "", err
	}
	h := sha256.New()
	// The manifest includes content digests for weights, tokenizer, configuration
	// and draft, so replacing a model under the same tag changes this identity.
	if err := json.NewEncoder(h).Encode(m); err != nil {
		return "", "", err
	}
	fmt.Fprintln(h, c.draftLookahead, mlx.GPUIsAvailable(), envconfig.Var("MLX_ENABLE_TF32"))
	layout.writeTo(h)
	for _, layer := range c.caches {
		fmt.Fprintf(h, "%T\n", layer)
	}
	return runtimeID, hex.EncodeToString(h.Sum(nil)), nil
}
