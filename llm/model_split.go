package llm

import (
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/fs/gguf"
)

const (
	// llama.cpp's split filename format reserves five decimal digits for the count.
	maxSplitModelFiles = 99999
	// Split aliases use <OLLAMA_MODELS>/blobs/.ollama-split-<manifest12>-<random>/model-00001-of-00002.gguf, with one directory per load.
	// Hard links give llama-server canonical shard names for auto-discovery because it has no CLI option for passing shard paths explicitly.
	splitModelDirPrefix = ".ollama-split-"
)

type splitModelLaunch struct {
	modelPath      string
	draftModelPath string
	projectors     []string
	dirs           []string
}

func materializeSplitModels(modelFiles, projectors []string, config LlamaServerConfig) (splitModelLaunch, error) {
	if len(modelFiles) == 0 {
		return splitModelLaunch{}, errors.New("GGUF model has no files")
	}

	launch := splitModelLaunch{
		modelPath:      modelFiles[0],
		draftModelPath: config.DraftModelPath,
		projectors:     projectors,
	}
	modelPath, dir, err := materializeSplitModel(modelFiles[0], modelFiles[1:], config.ManifestDigest)
	if err != nil {
		return splitModelLaunch{}, err
	}
	launch.modelPath = modelPath
	if dir != "" {
		launch.dirs = append(launch.dirs, dir)
		if len(projectors) > 0 && projectors[0] == modelFiles[0] {
			launch.projectors = slices.Clone(projectors)
			launch.projectors[0] = modelPath
		}
	}

	if config.DraftModelPath == "" {
		return launch, nil
	}
	draftPath, dir, err := materializeSplitModel(config.DraftModelPath, config.DraftModelShardPaths, config.ManifestDigest)
	if err != nil {
		_, cleanupErr := removeSplitModelDirs(launch.dirs)
		return splitModelLaunch{}, errors.Join(err, cleanupErr)
	}
	launch.draftModelPath = draftPath
	if dir != "" {
		launch.dirs = append(launch.dirs, dir)
	}
	return launch, nil
}

func materializeSplitModel(modelPath string, shardPaths []string, manifestDigest string) (string, string, error) {
	if len(shardPaths) == 0 {
		return modelPath, "", nil
	}

	paths := make([]string, 1, len(shardPaths)+1)
	paths[0] = modelPath
	paths = append(paths, shardPaths...)
	if len(paths) > maxSplitModelFiles {
		return "", "", fmt.Errorf("split GGUF has too many shards: %d", len(paths))
	}

	for _, path := range paths {
		info, err := os.Stat(path)
		if err != nil {
			return "", "", err
		}
		if !info.Mode().IsRegular() {
			return "", "", fmt.Errorf("split GGUF shard is not a regular file: %s", path)
		}
	}

	dir, err := os.MkdirTemp(filepath.Dir(modelPath), splitModelDirPattern(manifestDigest))
	if err != nil {
		return "", "", err
	}
	removeDir := true
	defer func() {
		if removeDir {
			_ = os.RemoveAll(dir)
		}
	}()

	for i, path := range paths {
		name := fmt.Sprintf("model-%05d-of-%05d.gguf", i+1, len(paths))
		if err := linkSplitModelFile(path, filepath.Join(dir, name)); err != nil {
			return "", "", fmt.Errorf("materialize split GGUF shard %d: %w", i+1, err)
		}
	}

	removeDir = false
	return filepath.Join(dir, fmt.Sprintf("model-%05d-of-%05d.gguf", 1, len(paths))), dir, nil
}

func splitModelDirPattern(manifestDigest string) string {
	if len(manifestDigest) == 64 {
		if _, err := hex.DecodeString(manifestDigest); err == nil {
			return splitModelDirPrefix + manifestDigest[:12] + "-"
		}
	}
	return splitModelDirPrefix
}

func removeSplitModelDirs(dirs []string) ([]string, error) {
	var errs []error
	remaining := dirs[:0]
	for _, dir := range dirs {
		if err := os.RemoveAll(dir); err != nil {
			errs = append(errs, err)
			remaining = append(remaining, dir)
		}
	}
	return remaining, errors.Join(errs...)
}

func (s *llamaServerRunner) removeSplitDirs() error {
	var err error
	s.splitDirs, err = removeSplitModelDirs(s.splitDirs)
	return err
}

func modelFileSize(path string, model *gguf.Model) uint64 {
	if size := model.FileSize(); size > 0 {
		return size
	}
	if info, err := os.Stat(path); err == nil && info.Size() >= 0 {
		return uint64(info.Size())
	}
	return 0
}

// PruneSplitModelDirs removes split-model alias directories older than before.
func PruneSplitModelDirs(root string, before time.Time) error {
	entries, err := os.ReadDir(root)
	if err != nil {
		return err
	}

	var errs []error
	for _, entry := range entries {
		if !entry.IsDir() || !strings.HasPrefix(entry.Name(), splitModelDirPrefix) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			errs = append(errs, err)
			continue
		}
		if !info.ModTime().Before(before) {
			continue
		}
		dir := filepath.Join(root, entry.Name())
		if err := os.RemoveAll(dir); err != nil {
			errs = append(errs, fmt.Errorf("remove split model directory %q: %w", dir, err))
		}
	}
	return errors.Join(errs...)
}

func linkSplitModelFile(src, dst string) error {
	linkErr := os.Link(src, dst)
	if linkErr == nil || runtime.GOOS != "windows" {
		return linkErr
	}

	in, err := os.Open(src)
	if err != nil {
		return errors.Join(linkErr, err)
	}
	defer in.Close()

	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return errors.Join(linkErr, err)
	}
	removeDst := true
	defer func() {
		if removeDst {
			_ = os.Remove(dst)
		}
	}()

	_, copyErr := io.Copy(out, in)
	closeErr := out.Close()
	if err := errors.Join(copyErr, closeErr); err != nil {
		return errors.Join(linkErr, err)
	}

	removeDst = false
	return nil
}
