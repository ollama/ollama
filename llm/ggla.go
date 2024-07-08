package llm

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"slices"
)

type containerGGLA struct {
	version uint32
}

func (c *containerGGLA) Name() string {
	return "ggla"
}

func (c *containerGGLA) Decode(rs io.ReadSeeker) (model, error) {
	slog.Info("decoding ggla")
	if err := binary.Read(rs, binary.LittleEndian, &c.version); err != nil {
		return nil, err
	}

	switch c.version {
	case 1:
	default:
		return nil, errors.New("invalid version")
	}

	model := newGGLA(c)
	err := model.decode(rs)
	return model, err
}

type ggla struct {
	*containerGGLA

	kv      KV
	tensors []*Tensor

	tensorOffset uint64
}

func newGGLA(container *containerGGLA) *ggla {
	return &ggla{
		containerGGLA: container,
		kv:            make(KV),
	}
}

func (llm *ggla) KV() KV {
	return llm.kv
}

func (llm *ggla) Tensors() Tensors {
	return Tensors{
		Items:  llm.tensors,
		Offset: llm.tensorOffset,
	}
}

func (llm *ggla) decode(rs io.ReadSeeker) error {
	var r uint32
	if err := binary.Read(rs, binary.LittleEndian, &r); err != nil {
		return err
	}
	llm.kv["r"] = r

	var alpha uint32
	if err := binary.Read(rs, binary.LittleEndian, &alpha); err != nil {
		return err
	}
	llm.kv["alpha"] = alpha

	offset, err := rs.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}

	llm.tensorOffset = uint64(offset)

	for {
		var dims uint32
		if err := binary.Read(rs, binary.LittleEndian, &dims); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return err
		}

		var namesize uint32
		if err := binary.Read(rs, binary.LittleEndian, &namesize); err != nil {
			return err
		}

		var t Tensor
		if err := binary.Read(rs, binary.LittleEndian, &t.Kind); err != nil {
			return err
		}

		t.Shape = make([]uint64, dims)
		for i := 0; uint32(i) < dims; i++ {
			var shape32 uint32
			if err := binary.Read(rs, binary.LittleEndian, &shape32); err != nil {
				return err
			}

			t.Shape[i] = uint64(shape32)
		}

		// ggla tensor shape is reversed
		// ref: https://github.com/ggerganov/llama.cpp/blob/29ae62d2ae163e2b68aa0ad3bf2ab4636de0c957/convert-lora-to-ggml.py#L44
		slices.Reverse(t.Shape)

		name := make([]byte, namesize)
		if err := binary.Read(rs, binary.LittleEndian, &name); err != nil {
			return err
		}

		t.Name = string(name)
		slog.Info(fmt.Sprintf("%s: [%d, %d] k=%d", t.Name, t.Shape[0], t.Shape[1], t.Kind))

		offset, err := rs.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}

		if _, err := rs.Seek((offset+31)&-32, io.SeekStart); err != nil {
			return err
		}

		offset, err = rs.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}

		t.Offset = uint64(offset)

		if _, err := rs.Seek(int64(t.Size()), io.SeekCurrent); err != nil {
			return err
		}

		llm.tensors = append(llm.tensors, &t)
	}
	return nil
}

func WriteGGLA(ws io.WriteSeeker, kv KV, ts []*Tensor) error {
	slog.Debug("writing ggla")
	if err := binary.Write(ws, binary.LittleEndian, []byte("algg")); err != nil {
		return err
	}

	if err := binary.Write(ws, binary.LittleEndian, uint32(1)); err != nil {
		return err
	}

	var r uint32
	var alpha uint32
	var ok bool

	if r, ok = kv["r"].(uint32); !ok {
		r = 8
	}

	if err := binary.Write(ws, binary.LittleEndian, r); err != nil {
		return err
	}

	if alpha, ok = kv["alpha"].(uint32); !ok {
		alpha = 16
	}

	if err := binary.Write(ws, binary.LittleEndian, alpha); err != nil {
		return err
	}

	for _, t := range ts {
		dims := 0
		for cnt := range len(t.Shape) {
			if t.Shape[cnt] > 0 {
				dims++
			}
		}

		if err := binary.Write(ws, binary.LittleEndian, uint32(dims)); err != nil {
			return err
		}

		if err := binary.Write(ws, binary.LittleEndian, uint32(len(t.Name))); err != nil {
			return err
		}

		if err := binary.Write(ws, binary.LittleEndian, t.Kind); err != nil {
			return err
		}

		for cnt := range dims {
			if err := binary.Write(ws, binary.LittleEndian, uint32(t.Shape[dims-1-cnt])); err != nil {
				return err
			}
		}

		if err := binary.Write(ws, binary.LittleEndian, []byte(t.Name)); err != nil {
			return err
		}

		offset, err := ws.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}

		var alignment int32 = 32
		pad := gglaPadding(int32(offset), alignment)
		if err := binary.Write(ws, binary.LittleEndian, bytes.Repeat([]byte{0}, int(pad))); err != nil {
			return err
		}

		if _, err := t.WriteTo(ws); err != nil {
			return err
		}
	}
	return nil
}

func gglaPadding(offset, align int32) int32 {
	return (align - offset%align) % align
}
