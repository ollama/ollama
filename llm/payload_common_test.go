package llm

import (
	"bytes"
	"compress/gzip"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExtractFile(t *testing.T) {
	const myContent = "Testing"

	asGzip := func(c string) []byte {
		var gzipBuffer bytes.Buffer
		gzipWriter := gzip.NewWriter(&gzipBuffer)
		_, err := gzipWriter.Write([]byte(myContent))
		require.NoError(t, err)
		_ = gzipWriter.Close()
		return gzipBuffer.Bytes()
	}

	specs := map[string]struct {
		sourceContent []byte
		destFileName  string
		expFileName   string
		expErr        bool
	}{
		"non gzip": {
			sourceContent: []byte(myContent),
			destFileName:  "testfile",
			expFileName:   "testfile",
		},
		"gzip": {
			sourceContent: asGzip(myContent),
			destFileName:  "testfile.gz",
			expFileName:   "testfile",
		},
		"invalid gzip": {
			sourceContent: []byte("not a gzip"),
			destFileName:  "testfile.gz",
			expErr:        true,
		},
	}
	for name, spec := range specs {
		t.Run(name, func(t *testing.T) {
			workDir := t.TempDir()
			targetFile := filepath.Join(workDir, spec.destFileName)
			gotErr := extractFile(bytes.NewReader(spec.sourceContent), targetFile)
			if spec.expErr {
				require.Error(t, gotErr)
				return
			}
			require.NoError(t, gotErr)
			expFileName := filepath.Join(workDir, spec.expFileName)
			require.FileExists(t, expFileName)
			file, err := os.Open(expFileName)
			defer file.Close()
			require.NoError(t, err)
			gotContent, err := io.ReadAll(file)
			require.NoError(t, err)
			assert.Equal(t, myContent, string(gotContent))
		})
	}
}
