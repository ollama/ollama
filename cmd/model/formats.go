package main

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/x/quant"
	"github.com/ollama/ollama/x/safetensors"
)

func (inv *inventory) tensorFile(ctx context.Context, a Artifact, role string) error {
	fi, err := inv.track(a.File)
	if err != nil {
		return err
	}
	f, err := os.Open(a.File)
	if err != nil {
		return err
	}
	var magic [4]byte
	_, err = io.ReadFull(f, magic[:])
	f.Close()
	if err != nil {
		return err
	}
	switch string(magic[:]) {
	case "GGUF":
		return inv.gguf(ctx, a, role, "little")
	case "FUGG":
		return inv.gguf(ctx, a, role, "big")
	default:
		return inv.safetensors(ctx, a, role, fi.Size())
	}
}

func (inv *inventory) safetensors(ctx context.Context, a Artifact, role string, fileSize int64) error {
	ext, err := safetensors.OpenForExtraction(a.File)
	if err != nil {
		return fmt.Errorf("%s: %w", a.File, err)
	}
	defer ext.Close()

	names := ext.ListTensors()
	if len(names) == 0 {
		return fmt.Errorf("empty safetensors file: %s", a.File)
	}
	metadata := ext.Metadata()
	tensors := make([]*Tensor, 0, len(names))
	var payloadSize int64
	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return err
		}
		data, err := ext.GetTensor(name)
		if err != nil {
			return fmt.Errorf("%s: %w", a.File, err)
		}
		if name == "" || data.Shape == nil {
			return fmt.Errorf("tensor %q has an empty name or missing shape", name)
		}
		dtype := strings.ToUpper(data.Dtype)
		shape := make([]uint64, len(data.Shape))
		for i, dim := range data.Shape {
			if dim < 0 {
				return fmt.Errorf("tensor %q has a negative dimension", name)
			}
			shape[i] = uint64(dim)
		}
		if data.Size < 0 {
			return fmt.Errorf("tensor %q has a negative payload size", name)
		}
		if payloadSize > math.MaxInt64-data.Size {
			return fmt.Errorf("safetensors payload size overflows")
		}
		payloadSize += data.Size
		tensors = append(tensors, &Tensor{Name: name, Role: role, DType: dtype, Shape: shape, Bytes: data.Size, Format: "safetensors", MediaType: a.MediaType, ByteOrder: "little", File: a.File, Layer: a.Name, Blob: a.Blob, Offset: data.Offset(), Metadata: metadata})
	}
	if payloadSize > fileSize {
		return fmt.Errorf("safetensors payload exceeds file size")
	}
	slices.SortFunc(tensors, func(a, b *Tensor) int {
		return cmp.Or(cmp.Compare(a.Offset, b.Offset), cmp.Compare(a.Bytes, b.Bytes))
	})
	end := fileSize - payloadSize
	for _, tensor := range tensors {
		if tensor.Offset != end {
			return fmt.Errorf("safetensors payload has a gap or overlap at offset %d", end)
		}
		end += tensor.Bytes
		if err := inv.addTensor(tensor); err != nil {
			return err
		}
	}
	if end != fileSize {
		return fmt.Errorf("safetensors payload has %d unindexed bytes", fileSize-end)
	}
	return nil
}

func (inv *inventory) gguf(ctx context.Context, a Artifact, role, byteOrder string) error {
	f, err := os.Open(a.File)
	if err != nil {
		return err
	}
	defer f.Close()
	model, err := ggml.Decode(f, -1)
	if err != nil {
		return fmt.Errorf("%s: %w", a.File, err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	meta := make(map[string]any, model.KV().Len())
	for key, value := range model.KV() {
		raw, err := json.Marshal(value)
		if err != nil {
			return fmt.Errorf("encode GGUF metadata %q: %w", key, err)
		}
		meta[key], err = decodeJSON(raw)
		if err != nil {
			return fmt.Errorf("decode GGUF metadata %q: %w", key, err)
		}
	}
	// split.* is a top-level GGUF namespace, so inspect the raw decoded key.
	if count, ok := meta["split.count"]; ok {
		n, ok := count.(json.Number)
		if !ok {
			return fmt.Errorf("invalid GGUF split.count: %v", count)
		}
		parts, err := n.Int64()
		if err != nil || parts < 1 {
			return fmt.Errorf("invalid GGUF split.count: %v", count)
		}
		if parts > 1 {
			return fmt.Errorf("split GGUF is not yet supported; refusing a partial comparison: %s", a.File)
		}
	}
	// Alignment is container layout. Decode derives parameter count from the
	// tensor inventory, so omit both values from semantic metadata comparison.
	delete(meta, "general.alignment")
	delete(meta, "general.parameter_count")
	key := role + ".gguf"
	if _, ok := inv.metadata[key]; ok {
		return fmt.Errorf("multiple GGUF files for role %q are ambiguous", role)
	}
	inv.metadata[key] = meta
	tensorSet := model.Tensors()
	for _, ti := range tensorSet.Items() {
		if err := ctx.Err(); err != nil {
			return err
		}
		if ti.Offset > math.MaxUint64-tensorSet.Offset {
			return fmt.Errorf("tensor %q offset overflows", ti.Name)
		}
		offset := tensorSet.Offset + ti.Offset
		bytes := ti.Size()
		if offset > math.MaxInt64 || bytes > math.MaxInt64 {
			return fmt.Errorf("tensor %q range overflows", ti.Name)
		}
		t := &Tensor{Name: ti.Name, Role: role, DType: ti.Type(), Shape: ti.Shape, Bytes: int64(bytes), Offset: int64(offset), Format: "gguf", MediaType: a.MediaType, ByteOrder: byteOrder, File: a.File, Layer: a.Name, Blob: a.Blob}
		if err := inv.addTensor(t); err != nil {
			return err
		}
	}
	return nil
}

func (inv *inventory) addTensor(t *Tensor) error {
	key := tensorKey(t.Role, t.Name)
	if _, ok := inv.tensors[key]; ok {
		return fmt.Errorf("duplicate tensor %q in role %q", t.Name, t.Role)
	}
	if t.Name == "" {
		return fmt.Errorf("empty tensor name in %s", t.File)
	}
	t.Elements = 1
	for _, dim := range t.Shape {
		if dim > 0 && t.Elements > math.MaxUint64/dim {
			return fmt.Errorf("tensor %q element count overflows", t.Name)
		}
		t.Elements *= dim
	}
	t.ModelDType = t.DType
	inv.tensors[key] = t
	return nil
}

func (inv *inventory) linkCompanions() error {
	for _, key := range unionKeys(inv.tensors, nil) {
		t := inv.tensors[key]
		if t.Format != "safetensors" || t.DType != "U32" {
			continue
		}
		// A suffix only establishes a relationship to an existing packed weight;
		// standalone biases are never skipped or guessed to be quantization data.
		for _, name := range []string{t.Name + ".scale", t.Name + ".bias", t.Name + ".global_scale", strings.TrimSuffix(t.Name, ".weight") + ".scales", strings.TrimSuffix(t.Name, ".weight") + ".biases"} {
			if _, ok := inv.tensors[tensorKey(t.Role, name)]; ok {
				t.Companions = append(t.Companions, name)
			}
		}
		slices.Sort(t.Companions)
		rawQuantType := strings.TrimSpace(t.Metadata["quant_type"])
		qt := quant.Canonical(rawQuantType)
		if qt == "" {
			if rawQuantType != "" || len(t.Companions) > 0 {
				t.ModelDType = "UNKNOWN QUANTIZATION"
				if rawQuantType != "" {
					t.ModelDType += " (" + rawQuantType + ")"
				}
				if err := inv.markCompanions(t); err != nil {
					return err
				}
			}
			continue
		}
		t.ModelDType = strings.ToUpper(qt)
		if err := inv.markCompanions(t); err != nil {
			return err
		}
		if len(t.Shape) == 0 {
			continue
		}
		group, bits, _ := quant.Params(qt) // guarded by Canonical: never use its unknown-format fallback
		if g := t.Metadata["group_size"]; g != "" {
			var err error
			group, err = strconv.Atoi(g)
			if err != nil || group <= 0 {
				return fmt.Errorf("tensor %q has invalid group_size %q", t.Name, g)
			}
		}
		shape := slices.Clone(t.Shape)
		factor := uint64(quant.PackFactor(qt))
		if shape[len(shape)-1] > math.MaxUint64/factor {
			return fmt.Errorf("tensor %q logical shape overflows", t.Name)
		}
		shape[len(shape)-1] *= factor
		t.Quantization = &Quantization{Type: qt, Bits: bits, GroupSize: group, LogicalShape: shape}
	}
	return nil
}

func (inv *inventory) markCompanions(parent *Tensor) error {
	for _, name := range parent.Companions {
		companion := inv.tensors[tensorKey(parent.Role, name)]
		if companion.CompanionOf != "" && companion.CompanionOf != parent.Name {
			return fmt.Errorf("tensor %q is a companion of both %q and %q", name, companion.CompanionOf, parent.Name)
		}
		companion.CompanionOf = parent.Name
		companion.ModelDType = parent.ModelDType
	}
	return nil
}
