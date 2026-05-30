package convert

import (
	"bufio"
	"encoding/binary"
	"io"
	"io/fs"
	"math"
	"os"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

const kindQ8_0 = 8

type q8Quantizer struct {
	src safetensor
}

func (q q8Quantizer) WriteTo(w io.Writer) (int64, error) {
	var elemSize int
	switch q.src.dtype {
	case "F16", "BF16":
		elemSize = 2
	case "F32":
		elemSize = 4
	default:
		return 0, io.ErrUnexpectedEOF
	}
	nElems := int(q.src.size) / elemSize

	f32s := make([]float32, nElems)

	if q.src.mmap != nil && len(q.src.mmap.data) > 0 && q.src.offset+q.src.size <= int64(len(q.src.mmap.data)) {
		data := q.src.mmap.data[q.src.offset : q.src.offset+q.src.size]
		decodeTensorToF32(q.src.dtype, data, f32s)
	} else {
		f, err := q.src.fs.Open(q.src.path)
		if err != nil {
			return 0, err
		}
		defer f.Close()

		r, err := func() (io.Reader, error) {
			if readerAt, ok := f.(io.ReaderAt); ok {
				return io.NewSectionReader(readerAt, q.src.offset, q.src.size), nil
			} else if seeker, ok := f.(io.Seeker); ok {
				_, err := seeker.Seek(q.src.offset, io.SeekStart)
				return f, err
			} else {
				_, err := io.CopyN(io.Discard, f, q.src.offset)
				return f, err
			}
		}()
		if err != nil {
			return 0, err
		}

		br := bufio.NewReaderSize(r, 128<<10)
		switch q.src.dtype {
		case "F16":
			u16s := make([]uint16, nElems)
			if err := binary.Read(br, binary.LittleEndian, u16s); err != nil {
				return 0, err
			}
			convertF16ToF32(f32s, u16s)
		case "BF16":
			u16s := make([]uint16, nElems)
			if err := binary.Read(br, binary.LittleEndian, u16s); err != nil {
				return 0, err
			}
			convertBF16ToF32(f32s, u16s)
		case "F32":
			if err := binary.Read(br, binary.LittleEndian, f32s); err != nil {
				return 0, err
			}
		}
	}

	out := quantizeQ8_0(f32s)
	n, err := w.Write(out)
	return int64(n), err
}

func decodeTensorToF32(dtype string, data []byte, dst []float32) {
	switch dtype {
	case "F16":
		u16s := make([]uint16, len(dst))
		for i := range u16s {
			u16s[i] = binary.LittleEndian.Uint16(data[i*2:])
		}
		convertF16ToF32(dst, u16s)
	case "BF16":
		u16s := make([]uint16, len(dst))
		for i := range u16s {
			u16s[i] = binary.LittleEndian.Uint16(data[i*2:])
		}
		convertBF16ToF32(dst, u16s)
	case "F32":
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
	}
}

func ConvertModelQ8_0(fsys fs.FS, f *os.File) error {
	kv, t, err := LoadModelMetadata(fsys)
	if err != nil {
		return err
	}
	conv := kv.(ModelConverter)

	ts, cleanup, err := parseTensors(fsys, strings.NewReplacer(conv.Replacements()...))
	if cleanup != nil {
		defer cleanup()
	}
	if err != nil {
		return err
	}

	ggmlTs := conv.Tensors(ts)
	for _, gt := range ggmlTs {
		if shouldQuantizeQ8(gt) {
			if st, ok := gt.WriterTo.(safetensor); ok {
				gt.WriterTo = q8Quantizer{src: st}
				gt.Kind = kindQ8_0
			}
		}
	}

	return writeFile(f, conv.KV(t), ggmlTs)
}

func shouldQuantizeQ8(t *ggml.Tensor) bool {
	return len(t.Shape) >= 2
}
