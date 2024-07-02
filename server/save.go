package server

import (
	"archive/tar"
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/ollama/ollama/types/model"
)

func saveModel(name model.Name, w io.Writer) error {
	manifest, err := ParseNamedManifest(name)
	if err != nil {
		return err
	}

	tw := tar.NewWriter(w)
	defer func() {
		if err := tw.Close(); err != nil {
			slog.Warn("failed to close tar writer", "error", err)
		}
	}()

	var index Index
	index.SchemaVersion = schemaVersion
	path, _ := strings.CutPrefix(manifest.filepath, modelsDir()+"/")
	index.Manifests = append(index.Manifests, path)

	var src []string
	src = append(src, manifest.filepath)
	for _, layer := range append(manifest.Layers, manifest.Config) {
		blob, err := GetBlobsPath(layer.Digest)
		if err != nil {
			return err
		}
		src = append(src, blob)
	}

	// Write manifest and blobs
	for _, filename := range src {
		fi, err := os.Stat(filename)
		if err != nil {
			return err
		}
		hdr, err := tar.FileInfoHeader(fi, "")
		if err != nil {
			return err
		}
		name, found := strings.CutPrefix(filename, modelsDir()+"/")
		if found {
			hdr.Name = name
		}
		err = tw.WriteHeader(hdr)
		if err != nil {
			return err
		}
		fs, err := os.Open(filename)
		if err != nil {
			return err
		}
		if _, err = io.Copy(tw, fs); err != nil {
			fs.Close()
			return err
		}
		fs.Close()
	}

	// Write index.json
	data, err := json.Marshal(index)
	if err != nil {
		return err
	}
	h := &tar.Header{
		Name:     "index.json",
		Typeflag: tar.TypeReg,
		ModTime:  time.Now(),
		Mode:     0644,
		Size:     int64(len(data)),
	}
	err = tw.WriteHeader(h)
	if err != nil {
		return err
	}
	_, err = tw.Write(data)
	if err != nil {
		return err
	}
	return nil
}
