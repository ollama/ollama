package convert

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"math"
	"slices"
	"strings"

)

type safetensorMetadata struct {
	Type    string   `json:"dtype"`
	Shape   []uint64 `json:"shape"`
	Offsets []int64  `json:"data_offsets"`
}

func parseSafetensors(fsys fs.FS, replacer *strings.Replacer, ps ...string) ([]Tensor, func(), error) {
	fp8Block, err := safetensorsFP8BlockSize(fsys)
	if err != nil {
		return nil, nil, err
	}

	mmaps, cleanup := tryMmapFiles(fsys, ps)

	var ts []Tensor
	for _, p := range ps {
		f, err := fsys.Open(p)
		if err != nil {
			cleanup()
			return nil, nil, err
		}
		defer f.Close()

		var n int64
		if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
			cleanup()
			return nil, nil, err
		}

		b := bytes.NewBuffer(make([]byte, 0, n))
		if _, err = io.CopyN(b, f, n); err != nil {
			cleanup()
			return nil, nil, err
		}

		var headers map[string]safetensorMetadata
		if err := json.NewDecoder(b).Decode(&headers); err != nil {
			cleanup()
			return nil, nil, err
		}

		keys := slices.Sorted(maps.Keys(headers))

		names := make(map[string]struct{}, len(keys))

		fp8Scales, err := collectSafetensorsFP8Scales(n, headers)
		if err != nil {
			cleanup()
			return nil, nil, err
		}

		for _, key := range keys {
			if value := headers[key]; value.Type != "" {
				if _, ok := fp8Scales.consumed[key]; ok {
					continue
				}

				// Scalar tensors (e.g. clipped linear min/max) are 0-dim in safetensors.
				// Promote them to 1-dim so they can be stored in GGUF.
				if len(value.Shape) == 0 {
					value.Shape = []uint64{1}
				}

				var scale *safetensorScale
				if value.Type == "F8_E4M3" {
					if !fp8Block.ok {
						cleanup()
						return nil, nil, fmt.Errorf("missing fp8 block size metadata for tensor %q", key)
					}
					scale = fp8Scales.byWeight[key]
					if scale == nil {
						cleanup()
						return nil, nil, fmt.Errorf("missing fp8 scale companion for tensor %q", key)
					}
				}

				ggufName := replacer.Replace(key)
				if _, ok := names[ggufName]; ok {
					cleanup()
					return nil, nil, fmt.Errorf("duplicate tensor name '%s' was found for this model", ggufName)
				}
				names[ggufName] = struct{}{}
				ts = append(ts, safetensor{
					fs:       fsys,
					path:     p,
					dtype:    value.Type,
					offset:   safetensorsPad(n, value.Offsets[0]),
					size:     safetensorsPad(n, value.Offsets[1]) - safetensorsPad(n, value.Offsets[0]),
					scale:    scale,
					fp8Block: fp8Block,
					mmap:     mmaps[p],
					tensorBase: &tensorBase{
						name:  ggufName,
						shape: value.Shape,
					},
				})
			}
		}
	}

	return ts, cleanup, nil
}

// safetensorsPad returns the padded size of the safetensors file given a length n and offset s
func safetensorsPad(n, offset int64) int64 {
	return 8 + n + offset
}

type safetensorScale struct {
	name   string
	dtype  string
	shape  []uint64
	offset int64
	size   int64
}

type safetensor struct {
	fs       fs.FS
	path     string
	dtype    string
	offset   int64
	size     int64
	scale    *safetensorScale
	fp8Block safetensorFP8BlockSize
	mmap     *mmapRegion
	*tensorBase
}

func (st safetensor) Kind() uint32 {
	kind := st.tensorBase.Kind()
	if st.dtype == "BF16" &&
		!strings.HasPrefix(st.name, "v.") &&
		!strings.HasPrefix(st.name, "s.") &&
		!strings.HasPrefix(st.name, "mm.") &&
		!strings.Contains(st.name, "ffn_gate_inp_shexp.weight") &&
		kind != tensorKindFP32 {
		kind = tensorKindBF16
	}
	if st.dtype == "F8_E4M3" && kind != tensorKindFP32 {
		kind = tensorKindBF16
	}

	return kind
}

func (st safetensor) SourceDType() string {
	return st.dtype
}

func (st safetensor) Clone() Tensor {
	return &safetensor{
		fs:       st.fs,
		path:     st.path,
		dtype:    st.dtype,
		offset:   st.offset,
		size:     st.size,
		scale:    st.scale.Clone(),
		fp8Block: st.fp8Block,
		mmap:     st.mmap,
		tensorBase: &tensorBase{
			name:     st.name,
			repacker: st.repacker,
			shape:    slices.Clone(st.shape),
		},
	}
}

func (ss *safetensorScale) Clone() *safetensorScale {
	if ss == nil {
		return nil
	}
	return &safetensorScale{
		name:   ss.name,
		dtype:  ss.dtype,
		shape:  slices.Clone(ss.shape),
		offset: ss.offset,
		size:   ss.size,
	}
}

func (st safetensor) WriteTo(w io.Writer) (int64, error) {
	passthrough := (st.repacker == nil) &&
		((st.dtype == "F32" && st.Kind() == tensorKindFP32) ||
			(st.dtype == "F16" && st.Kind() == tensorKindFP16) ||
			(st.dtype == "BF16" && st.Kind() == tensorKindBF16) ||
			(st.dtype == "U8"))

	if st.mmap != nil && len(st.mmap.data) > 0 && st.offset+st.size <= int64(len(st.mmap.data)) {
		return st.writeFromMmap(w, passthrough)
	}
	return st.writeFromFile(w, passthrough)
}

func (st safetensor) writeFromMmap(w io.Writer, passthrough bool) (int64, error) {
	data := st.mmap.data[st.offset : st.offset+st.size]

	if passthrough {
		n, err := w.Write(data)
		return int64(n), err
	}

	return st.writeFullBuffer(w, bytes.NewReader(data))
}

func (st safetensor) writeFromFile(w io.Writer, passthrough bool) (int64, error) {
	f, err := st.fs.Open(st.path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	r, err := func() (io.Reader, error) {
		if readerAt, ok := f.(io.ReaderAt); ok {
			return io.NewSectionReader(readerAt, st.offset, st.size), nil
		} else if seeker, ok := f.(io.Seeker); ok {
			_, err := seeker.Seek(st.offset, io.SeekStart)
			return f, err
		} else {
			_, err := io.CopyN(io.Discard, f, st.offset)
			return f, err
		}
	}()
	if err != nil {
		return 0, err
	}

	br := bufio.NewReaderSize(r, min(32<<10, int(st.size)))

	if passthrough {
		return io.CopyN(w, br, st.size)
	}

	return st.writeFullBuffer(w, br)
}

func writeOutputChunk(w io.Writer, f32s []float32, kind uint32) (int64, error) {
	switch kind {
	case tensorKindFP32:
		return int64(len(f32s) * 4), binary.Write(w, binary.LittleEndian, f32s)
	case tensorKindFP16:
		f16s := make([]uint16, len(f32s))
		convertF32ToF16(f16s, f32s)
		return int64(len(f16s) * 2), binary.Write(w, binary.LittleEndian, f16s)
	case tensorKindBF16:
		bf16s := make([]uint16, len(f32s))
		convertF32ToBF16(bf16s, f32s)
		return int64(len(bf16s) * 2), binary.Write(w, binary.LittleEndian, bf16s)
	default:
		return 0, fmt.Errorf("unknown storage type: %d", kind)
	}
}

func (st safetensor) writeFullBuffer(w io.Writer, r io.Reader) (int64, error) {
	br := bufio.NewReaderSize(r, min(32<<10, int(st.size)))

	var f32s []float32
	var err error
	switch st.dtype {
	case "F32":
		f32s = make([]float32, st.size/4)
		if err = binary.Read(br, binary.LittleEndian, f32s); err != nil {
			return 0, err
		}
	case "F16":
		u16s := make([]uint16, st.size/2)
		if err = binary.Read(br, binary.LittleEndian, u16s); err != nil {
			return 0, err
		}

		f32s = make([]float32, len(u16s))
		convertF16ToF32(f32s, u16s)
	case "BF16":
		u16s := make([]uint16, st.size/2)
		if err = binary.Read(br, binary.LittleEndian, u16s); err != nil {
			return 0, err
		}

		f32s = make([]float32, len(u16s))
		convertBF16ToF32(f32s, u16s)
	case "F8_E4M3":
		u8s := make([]uint8, st.size)
		if err = binary.Read(br, binary.LittleEndian, u8s); err != nil {
			return 0, err
		}

		f32s, err = st.decodeFP8E4M3(u8s)
		if err != nil {
			return 0, err
		}
	default:
		return 0, fmt.Errorf("unknown data type: %s", st.dtype)
	}

	if st.repacker != nil {
		f32s, err = st.repacker(st.Name(), f32s, st.Shape())
		if err != nil {
			return 0, err
		}
	}

	return writeOutputChunk(w, f32s, st.Kind())
}

type safetensorsFP8Scales struct {
	byWeight map[string]*safetensorScale
	consumed map[string]struct{}
}

func collectSafetensorsFP8Scales(n int64, headers map[string]safetensorMetadata) (safetensorsFP8Scales, error) {
	scales := safetensorsFP8Scales{
		byWeight: make(map[string]*safetensorScale),
		consumed: make(map[string]struct{}),
	}

	for key, value := range headers {
		if value.Type != "F8_E4M3" {
			continue
		}

		scaleKey, scaleValue, ok, err := safetensorsFP8Scale(key, headers)
		if err != nil {
			return safetensorsFP8Scales{}, err
		}
		if !ok {
			continue
		}
		if _, ok := scales.consumed[scaleKey]; ok {
			return safetensorsFP8Scales{}, fmt.Errorf("fp8 scale companion %q is used by multiple tensors", scaleKey)
		}

		scales.byWeight[key] = &safetensorScale{
			name:   scaleKey,
			dtype:  scaleValue.Type,
			shape:  slices.Clone(scaleValue.Shape),
			offset: safetensorsPad(n, scaleValue.Offsets[0]),
			size:   safetensorsPad(n, scaleValue.Offsets[1]) - safetensorsPad(n, scaleValue.Offsets[0]),
		}
		scales.consumed[scaleKey] = struct{}{}
	}

	return scales, nil
}

func safetensorsFP8Scale(key string, headers map[string]safetensorMetadata) (string, safetensorMetadata, bool, error) {
	candidates := safetensorsFP8ScaleCandidates(key)

	var scaleKey string
	var scaleValue safetensorMetadata
	if strings.HasSuffix(key, ".weight") {
		// Keep support for compressed-tensors exports that place the scale name
		// between the module path and weight suffix.
		base := strings.TrimSuffix(key, ".weight")
		candidates = appendUnique(candidates, base+".weight_scale")
		candidates = appendUnique(candidates, base+".weight_scale_inv")
	}

	for _, candidate := range candidates {
		if value, ok := headers[candidate]; ok && value.Type != "" {
			if scaleKey != "" {
				return "", safetensorMetadata{}, false, fmt.Errorf("multiple fp8 scale companions for tensor %q: %q and %q", key, scaleKey, candidate)
			}
			scaleKey = candidate
			scaleValue = value
		}
	}
	if scaleKey == "" {
		return "", safetensorMetadata{}, false, nil
	}

	return scaleKey, scaleValue, true, nil
}

func safetensorsFP8ScaleCandidates(key string) []string {
	var candidates []string
	candidates = appendUnique(candidates, key+"_scale")
	candidates = appendUnique(candidates, key+"_scale_inv")
	candidates = appendUnique(candidates, key+".scale")
	candidates = appendUnique(candidates, key+".scale_inv")
	return candidates
}

func appendUnique(values []string, value string) []string {
	if !slices.Contains(values, value) {
		values = append(values, value)
	}
	return values
}

type safetensorFP8BlockSize struct {
	rows int
	cols int
	ok   bool
}

type safetensorsSourceQuantization struct {
	QuantMethod     string `json:"quant_method"`
	Format          string `json:"format"`
	WeightBlockSize []int  `json:"weight_block_size"`
	ConfigGroups    map[string]struct {
		Format  string `json:"format"`
		Weights struct {
			BlockStructure []int  `json:"block_structure"`
			NumBits        int    `json:"num_bits"`
			Type           string `json:"type"`
		} `json:"weights"`
	} `json:"config_groups"`
}

type safetensorsModelConfig struct {
	Quantization       safetensorsSourceQuantization `json:"quantization"`
	QuantizationConfig safetensorsSourceQuantization `json:"quantization_config"`
	CompressionConfig  safetensorsSourceQuantization `json:"compression_config"`
	TextConfig         struct {
		Quantization       safetensorsSourceQuantization `json:"quantization"`
		QuantizationConfig safetensorsSourceQuantization `json:"quantization_config"`
		CompressionConfig  safetensorsSourceQuantization `json:"compression_config"`
	} `json:"text_config"`
}

func safetensorsFP8BlockSize(fsys fs.FS) (safetensorFP8BlockSize, error) {
	bts, err := fs.ReadFile(fsys, "config.json")
	if errors.Is(err, fs.ErrNotExist) {
		return safetensorFP8BlockSize{}, nil
	}
	if err != nil {
		return safetensorFP8BlockSize{}, err
	}
	bts = sanitizeNonFiniteJSON(bts)

	var cfg safetensorsModelConfig
	if err := json.Unmarshal(bts, &cfg); err != nil {
		return safetensorFP8BlockSize{}, fmt.Errorf("parse config.json fp8 metadata: %w", err)
	}

	var blocks []safetensorFP8BlockSize
	for _, q := range []safetensorsSourceQuantization{
		cfg.Quantization,
		cfg.QuantizationConfig,
		cfg.CompressionConfig,
		cfg.TextConfig.Quantization,
		cfg.TextConfig.QuantizationConfig,
		cfg.TextConfig.CompressionConfig,
	} {
		if strings.EqualFold(q.QuantMethod, "fp8") && len(q.WeightBlockSize) == 2 {
			block, err := newSafetensorFP8BlockSize(q.WeightBlockSize[0], q.WeightBlockSize[1])
			if err != nil {
				return safetensorFP8BlockSize{}, err
			}
			blocks = append(blocks, block)
		}

		if !strings.EqualFold(q.QuantMethod, "compressed-tensors") && !strings.EqualFold(q.Format, "float-quantized") {
			continue
		}
		for _, group := range q.ConfigGroups {
			if !strings.EqualFold(group.Format, "float-quantized") ||
				group.Weights.NumBits != 8 ||
				!strings.EqualFold(group.Weights.Type, "float") ||
				len(group.Weights.BlockStructure) != 2 {
				continue
			}
			block, err := newSafetensorFP8BlockSize(group.Weights.BlockStructure[0], group.Weights.BlockStructure[1])
			if err != nil {
				return safetensorFP8BlockSize{}, err
			}
			blocks = append(blocks, block)
		}
	}

	if len(blocks) == 0 {
		return safetensorFP8BlockSize{}, nil
	}

	block := blocks[0]
	for _, other := range blocks[1:] {
		if other.rows != block.rows || other.cols != block.cols {
			return safetensorFP8BlockSize{}, fmt.Errorf("multiple fp8 block sizes in config.json: %dx%d and %dx%d", block.rows, block.cols, other.rows, other.cols)
		}
	}
	return block, nil
}

func newSafetensorFP8BlockSize(rows, cols int) (safetensorFP8BlockSize, error) {
	if rows <= 0 || cols <= 0 {
		return safetensorFP8BlockSize{}, fmt.Errorf("invalid fp8 block size %dx%d", rows, cols)
	}
	return safetensorFP8BlockSize{rows: rows, cols: cols, ok: true}, nil
}

func (st safetensor) decodeFP8E4M3(data []byte) ([]float32, error) {
	if st.scale == nil {
		return nil, fmt.Errorf("missing fp8 scale companion for tensor %q", st.name)
	}
	if !st.fp8Block.ok {
		return nil, fmt.Errorf("missing fp8 block size metadata for tensor %q", st.name)
	}
	if len(st.shape) != 2 {
		return nil, fmt.Errorf("expected 2D fp8 tensor %q, got shape %v", st.name, st.shape)
	}

	rows, cols := int(st.shape[0]), int(st.shape[1])
	if rows < 0 || cols < 0 || rows*cols != len(data) {
		return nil, fmt.Errorf("fp8 tensor %q shape %v does not match %d bytes", st.name, st.shape, len(data))
	}

	scale, err := st.readScale()
	if err != nil {
		return nil, err
	}

	if len(st.scale.shape) != 2 {
		return nil, fmt.Errorf("expected 2D fp8 scale tensor %q, got shape %v", st.scale.name, st.scale.shape)
	}

	blockRows := st.fp8Block.rows
	blockCols := st.fp8Block.cols
	scaleRows, scaleCols := int(st.scale.shape[0]), int(st.scale.shape[1])
	expectedRows := (rows + blockRows - 1) / blockRows
	expectedCols := (cols + blockCols - 1) / blockCols
	if scaleRows != expectedRows || scaleCols != expectedCols {
		return nil, fmt.Errorf("unexpected fp8 scale shape %v for tensor %q shape %v; want [%d %d]", st.scale.shape, st.name, st.shape, expectedRows, expectedCols)
	}
	if len(scale) != scaleRows*scaleCols {
		return nil, fmt.Errorf("fp8 scale tensor %q shape %v does not match decoded length %d", st.scale.name, st.scale.shape, len(scale))
	}

	f32s := make([]float32, len(data))
	for r := range rows {
		scaleRow := r / blockRows
		rowOffset := r * cols
		for c := range cols {
			f32s[rowOffset+c] = decodeFloat8E4M3FN(data[rowOffset+c]) * scale[scaleRow*scaleCols+c/blockCols]
		}
	}

	return f32s, nil
}

func (st safetensor) readScale() ([]float32, error) {
	var br *bufio.Reader
	if st.mmap != nil && len(st.mmap.data) > 0 && st.scale.offset+st.scale.size <= int64(len(st.mmap.data)) {
		data := st.mmap.data[st.scale.offset : st.scale.offset+st.scale.size]
		br = bufio.NewReaderSize(bytes.NewReader(data), min(32<<10, int(st.scale.size)))
	} else {
		r, err := st.sectionReader(st.scale.offset, st.scale.size)
		if err != nil {
			return nil, fmt.Errorf("failed to read fp8 scale tensor %q: %w", st.scale.name, err)
		}
		if closer, ok := r.(io.Closer); ok {
			defer closer.Close()
		}
		br = bufio.NewReaderSize(r, min(32<<10, int(st.scale.size)))
	}

	switch st.scale.dtype {
	case "F32":
		f32s := make([]float32, st.scale.size/4)
		if err := binary.Read(br, binary.LittleEndian, f32s); err != nil {
			return nil, err
		}
		return f32s, nil
	case "F16":
		u16s := make([]uint16, st.scale.size/2)
		if err := binary.Read(br, binary.LittleEndian, u16s); err != nil {
			return nil, err
		}
		f32s := make([]float32, len(u16s))
		convertF16ToF32(f32s, u16s)
		return f32s, nil
	case "BF16":
		u16s := make([]uint16, st.scale.size/2)
		if err := binary.Read(br, binary.LittleEndian, u16s); err != nil {
			return nil, err
		}
		f32s := make([]float32, len(u16s))
		convertBF16ToF32(f32s, u16s)
		return f32s, nil
	default:
		return nil, fmt.Errorf("unsupported fp8 scale dtype %q for tensor %q", st.scale.dtype, st.scale.name)
	}
}

func (st safetensor) sectionReader(offset, size int64) (io.Reader, error) {
	f, err := st.fs.Open(st.path)
	if err != nil {
		return nil, err
	}

	if readerAt, ok := f.(io.ReaderAt); ok {
		return &readCloserReader{
			Reader: io.NewSectionReader(readerAt, offset, size),
			Closer: f,
		}, nil
	}
	if seeker, ok := f.(io.Seeker); ok {
		if _, err := seeker.Seek(offset, io.SeekStart); err != nil {
			f.Close()
			return nil, err
		}
		return &readCloserReader{
			Reader: io.LimitReader(f, size),
			Closer: f,
		}, nil
	}
	if _, err := io.CopyN(io.Discard, f, offset); err != nil {
		f.Close()
		return nil, err
	}
	return &readCloserReader{
		Reader: io.LimitReader(f, size),
		Closer: f,
	}, nil
}

type readCloserReader struct {
	io.Reader
	io.Closer
}

func decodeFloat8E4M3FN(v byte) float32 {
	sign := float32(1)
	if v&0x80 != 0 {
		sign = -1
	}

	exp := int((v >> 3) & 0x0f)
	mant := int(v & 0x07)
	if exp == 0 {
		if mant == 0 {
			return 0 * sign
		}
		return sign * float32(math.Ldexp(float64(mant)/8, -6))
	}
	if exp == 0x0f && mant == 0x07 {
		return float32(math.NaN())
	}

	return sign * float32(math.Ldexp(1+float64(mant)/8, exp-7))
}
