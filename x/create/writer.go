package create

import (
	"bytes"
	"fmt"
	"io"
	"path/filepath"

	"github.com/ollama/ollama/x/safetensors"
)

const mediaTypeImageTensor = "application/vnd.ollama.image.tensor"

// BlobStore stores a finished blob and returns its layer info. The writer
// produces the blob bytes; where they are stored — the local model store, or a
// remote target in a future networked create — is the store's concern.
type BlobStore interface {
	WriteBlob(r io.Reader, mediaType, name string) (LayerInfo, error)
}

// WriteBlobs executes a plan's blobs: for each blob it resolves the tensors'
// sources, produces the blob bytes, and stores the result.
func WriteBlobs(specs []BlobSpec, modelDir string, store BlobStore) ([]LayerInfo, error) {
	src := newSourceFiles(modelDir)
	defer src.close()

	layers := make([]LayerInfo, 0, len(specs))
	for _, spec := range specs {
		layer, err := writeBlob(spec, src, store)
		if err != nil {
			return nil, err
		}
		layers = append(layers, layer)
	}
	return layers, nil
}

// writeBlob resolves each tensor's sources and produces the blob.
func writeBlob(spec BlobSpec, src *sourceFiles, store BlobStore) (LayerInfo, error) {
	needsMLX := blobNeedsMLX(spec)
	var (
		tensors []*safetensors.TensorData
		items   []quantizeItem
	)
	for _, ts := range spec.Tensors {
		sources, err := src.resolve(ts.Sources)
		if err != nil {
			return LayerInfo{}, err
		}
		if needsMLX {
			reader, err := quantizeInputReader(ts, sources)
			if err != nil {
				return LayerInfo{}, fmt.Errorf("blob %s: tensor %s: %w", spec.Name, ts.Name, err)
			}
			items = append(items, quantizeItem{name: ts.Name, quantize: ts.Quantize, reader: reader, decodeFP8: needsFP8Decode(ts.Transform)})
		} else {
			td, err := applyByteTransform(ts, sources)
			if err != nil {
				return LayerInfo{}, fmt.Errorf("blob %s: tensor %s: %w", spec.Name, ts.Name, err)
			}
			tensors = append(tensors, td)
		}
	}

	// The quantizer computes the blob's quant metadata itself and ignores
	// spec.Metadata; today only prequant blobs carry Metadata and they never
	// take the MLX path.
	var r io.Reader
	if needsMLX {
		blobData, err := quantizeBlob(items)
		if err != nil {
			return LayerInfo{}, fmt.Errorf("quantize blob %s: %w", spec.Name, err)
		}
		r = bytes.NewReader(blobData)
	} else {
		r = safetensors.BuildPackedSafetensorsReaderWithMetadata(tensors, spec.Metadata)
	}

	layer, err := store.WriteBlob(r, mediaTypeImageTensor, spec.Name)
	if err != nil {
		return LayerInfo{}, fmt.Errorf("write blob %s: %w", spec.Name, err)
	}
	return layer, nil
}

// quantizeInputReader builds the safetensors-wrapped reader the quantizer
// consumes for one tensor: a paired weight+scale for FP8 decode, a byte-stacked
// 3D tensor for experts, or the tensor itself. The output tensor is always
// keyed by ts.Name so the quantizer can look it up by name.
func quantizeInputReader(ts TensorSpec, sources []*safetensors.TensorData) (io.Reader, error) {
	switch ts.Transform {
	case TransformNone:
		if len(sources) != 1 {
			return nil, fmt.Errorf("transform none expects 1 source, got %d", len(sources))
		}
		return sources[0].WithName(ts.Name).SafetensorsReader(), nil
	case TransformDecodeFP8:
		if len(sources) != 2 {
			return nil, fmt.Errorf("transform decode_fp8 expects weight+scale, got %d sources", len(sources))
		}
		return buildSourceFP8Reader(sources[0].WithName(ts.Name), sources[1]), nil
	case TransformDecodeStackFP8:
		if len(sources) == 0 || len(sources)%2 != 0 {
			return nil, fmt.Errorf("transform decode_stack_fp8 expects N weights followed by N scales, got %d sources", len(sources))
		}
		n := len(sources) / 2
		weights, scales := sources[:n], sources[n:]
		stackedWeight, err := stackExpertTensors(ts.Name, ts.OutDtype, ts.OutShape, weights)
		if err != nil {
			return nil, err
		}
		scaleShape := append([]int32{int32(n)}, scales[0].Shape...)
		stackedScale, err := stackExpertTensors(ts.Name+".scale_inv", scales[0].Dtype, scaleShape, scales)
		if err != nil {
			return nil, err
		}
		return safetensors.BuildPackedSafetensorsReader([]*safetensors.TensorData{stackedWeight, stackedScale}), nil
	case TransformStackExperts:
		stacked, err := stackExpertTensors(ts.Name, ts.OutDtype, ts.OutShape, sources)
		if err != nil {
			return nil, err
		}
		return stacked.SafetensorsReader(), nil
	default:
		return nil, fmt.Errorf("transform %q is not supported for a quantized tensor", ts.Transform)
	}
}

// needsFP8Decode reports whether a transform dequantizes block-FP8 weights on
// the MLX thread (a single tensor or a stacked per-expert group).
func needsFP8Decode(t Transform) bool {
	return t == TransformDecodeFP8 || t == TransformDecodeStackFP8
}

// blobNeedsMLX reports whether any of the blob's tensors require the MLX path
// — either a quantization target or an FP8 decode (which cannot be done with
// byte operations alone).
func blobNeedsMLX(spec BlobSpec) bool {
	for _, ts := range spec.Tensors {
		if ts.Quantize != "" || needsFP8Decode(ts.Transform) {
			return true
		}
	}
	return false
}

// sourceFiles opens and caches the source safetensors files so each shard is
// opened once across all the blobs that read from it.
type sourceFiles struct {
	dir   string
	cache map[string]*safetensors.TensorExtractor
}

func newSourceFiles(dir string) *sourceFiles {
	return &sourceFiles{dir: dir, cache: make(map[string]*safetensors.TensorExtractor)}
}

func (s *sourceFiles) resolve(sources []SourceTensor) ([]*safetensors.TensorData, error) {
	out := make([]*safetensors.TensorData, len(sources))
	for i, st := range sources {
		ext, err := s.extractor(st.File)
		if err != nil {
			return nil, err
		}
		td, err := ext.GetTensor(st.Name)
		if err != nil {
			return nil, fmt.Errorf("read %s from %s: %w", st.Name, st.File, err)
		}
		out[i] = td
	}
	return out, nil
}

func (s *sourceFiles) extractor(file string) (*safetensors.TensorExtractor, error) {
	if ext, ok := s.cache[file]; ok {
		return ext, nil
	}
	ext, err := safetensors.OpenForExtraction(filepath.Join(s.dir, file))
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", file, err)
	}
	s.cache[file] = ext
	return ext, nil
}

func (s *sourceFiles) close() {
	for _, ext := range s.cache {
		ext.Close()
	}
}
