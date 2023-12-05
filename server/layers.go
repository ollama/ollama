package server

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"

	"golang.org/x/exp/slices"
)

type Layers struct {
	items []*Layer
}

func (ls *Layers) Add(layer *Layer) {
	if layer.Size > 0 {
		ls.items = append(ls.items, layer)
	}
}

func (ls *Layers) Replace(layer *Layer) {
	if layer.Size > 0 {
		mediatype := layer.MediaType
		layers := slices.DeleteFunc(ls.items, func(l *Layer) bool {
			return l.MediaType == mediatype
		})

		ls.items = append(layers, layer)
	}
}

type Layer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
	From      string `json:"from,omitempty"`

	tempFileName string
}

func NewLayer(r io.Reader, mediatype string) (*Layer, error) {
	blobs, err := GetBlobsPath("")
	if err != nil {
		return nil, err
	}

	delimiter := ":"
	if runtime.GOOS == "windows" {
		delimiter = "-"
	}

	pattern := strings.Join([]string{"sha256", "*-partial"}, delimiter)
	temp, err := os.CreateTemp(blobs, pattern)
	if err != nil {
		return nil, err
	}
	defer temp.Close()

	sha256sum := sha256.New()
	n, err := io.Copy(io.MultiWriter(temp, sha256sum), r)
	if err != nil {
		return nil, err
	}

	return &Layer{
		MediaType:    mediatype,
		Digest:       fmt.Sprintf("sha256:%x", sha256sum.Sum(nil)),
		Size:         n,
		tempFileName: temp.Name(),
	}, nil
}

func NewLayerFromLayer(digest, mediatype, from string) (*Layer, error) {
	blob, err := GetBlobsPath(digest)
	if err != nil {
		return nil, err
	}

	fi, err := os.Stat(blob)
	if err != nil {
		return nil, err
	}

	return &Layer{
		MediaType: mediatype,
		Digest:    digest,
		Size:      fi.Size(),
		From:      from,
	}, nil
}

func (l *Layer) Commit() (bool, error) {
	// always remove temp
	defer os.Remove(l.tempFileName)

	blob, err := GetBlobsPath(l.Digest)
	if err != nil {
		return false, err
	}

	if _, err := os.Stat(blob); err != nil {
		return true, os.Rename(l.tempFileName, blob)
	}

	return false, nil
}
