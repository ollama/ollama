package server

import (
	"archive/tar"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/types/model"
)

func UnTar(dst string, src io.Reader) error {
	tr := tar.NewReader(src)
	for {
		hdr, err := tr.Next()

		switch {
		case err == io.EOF:
			return nil
		case err != nil:
			return err
		case hdr == nil:
			continue
		}

		dstFileDir := filepath.Join(dst, hdr.Name)

		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(dstFileDir, 0775); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(dstFileDir), 0755); err != nil && !errors.Is(err, os.ErrExist) {
				return err
			}
			file, err := os.OpenFile(dstFileDir, os.O_CREATE|os.O_RDWR, os.FileMode(hdr.Mode))
			if err != nil {
				return err
			}
			_, err = io.Copy(file, tr)
			if err != nil {
				file.Close()
				return err
			}
			file.Close()
		}
	}
}

func loadBlobByDigest(srcDir, srcDigest string) error {
	srcBlobName := strings.ReplaceAll(srcDigest, ":", "-")
	dstBlobDir, err := GetBlobsPath("")
	if err != nil {
		return err
	}

	dstBlobPath := filepath.Join(dstBlobDir, srcBlobName)
	if _, err = os.Stat(dstBlobPath); err == nil {
		// Blob already exist, just return
		return nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return err
	}

	dstFile, err := os.OpenFile(dstBlobPath, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer dstFile.Close()

	srcBlobPath := filepath.Join(srcDir, "blobs/"+srcBlobName)
	srcFile, err := os.Open(srcBlobPath)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	sha256sum := sha256.New()
	_, err = io.Copy(io.MultiWriter(dstFile, sha256sum), srcFile)
	if err != nil {
		return err
	}

	if srcDigest != fmt.Sprintf("sha256:%x", sha256sum.Sum(nil)) {
		return fmt.Errorf("Failed to load layer, digest mismatch, expected %q, got %q", srcDigest, fmt.Sprintf("sha256:%x", sha256sum.Sum(nil)))
	}

	return nil
}

func loadManifest(srcManifestPath, dstManifestPath string) error {
	_, err := os.Stat(dstManifestPath)
	if err == nil {
		// Manifest already exist
		return nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(dstManifestPath), 0755); err != nil && !errors.Is(err, os.ErrExist) {
		return err
	}
	dst, err := os.OpenFile(dstManifestPath, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return err
	}
	defer dst.Close()

	src, err := os.Open(srcManifestPath)
	if err != nil {
		return err
	}
	defer src.Close()

	_, err = io.Copy(dst, src)
	return err
}

func importModal(tmpDir string) (string, error) {
	indexPath := filepath.Join(tmpDir, "index.json")
	fi, err := os.Open(indexPath)
	if err != nil {
		return "", err
	}
	defer fi.Close()

	var idx Index
	err = json.NewDecoder(fi).Decode(&idx)
	if err != nil {
		return "", err
	}

	if idx.SchemaVersion != schemaVersion {
		return "", fmt.Errorf("unsupported load schema version %v", idx.SchemaVersion)
	}

	// We only support single manifest for now
	manifestPath := idx.Manifests[0]
	srcManifestPath := filepath.Join(tmpDir, manifestPath)
	mfi, err := os.Open(srcManifestPath)
	if err != nil {
		return "", err
	}
	defer mfi.Close()

	var manifest ManifestV2
	if err = json.NewDecoder(mfi).Decode(&manifest); err != nil {
		return "", err
	}

	err = loadManifest(srcManifestPath, filepath.Join(modelsDir(), manifestPath))
	if err != nil {
		return "", err
	}

	err = loadBlobByDigest(tmpDir, manifest.Config.Digest)
	if err != nil {
		return "", err
	}
	for _, layer := range manifest.Layers {
		err = loadBlobByDigest(tmpDir, layer.Digest)
		if err != nil {
			return "", err
		}
	}
	manifestPath, _ = strings.CutPrefix(manifestPath, "manifests/")
	mpName := model.ParseNameFromFilepath(manifestPath)
	return mpName.DisplayShortest(), nil
}

func loadModel(req io.Reader) (string, error) {
	tempDir, err := os.MkdirTemp("", "ollama-import-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(tempDir)

	err = UnTar(tempDir, req)
	if err != nil {
		return "", err
	}

	loadedManifest, err := importModal(tempDir)
	if err != nil {
		// clean up unused layers and manifests if fail to load
		if err2 := PruneLayers(); err2 != nil {
			slog.Warn("Failed to prune layers", "error", err)
		}

		manifestsPath, err2 := GetManifestPath()
		if err2 != nil {
			slog.Warn("Failed to get manifest", "error", err)
		}

		if err2 := PruneDirectory(manifestsPath); err2 != nil {
			slog.Warn("Failed to prune manifest directory", "error", err)
		}
		return "", err
	}
	return loadedManifest, nil
}
