package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"hash"
	"io"
	"math"
	"os"
	"slices"
	"strings"
)

type StatsReport struct {
	Left           QuantStats
	Right          QuantStats
	Comparisons    []NumericComparison
	BytesRead      int64
	ExtraBytesRead int64
}

type QuantStats struct {
	MXFP8Tensors int
	MXFP4Tensors int
	NVFP4Tensors int

	E4M3PayloadBlocks   uint64
	E4M3SaturatedBlocks uint64
	E8M0Scales          uint64
	E8M0MaxScales       uint64
	E8M0InvalidScales   uint64
	E4M3Scales          uint64
	E4M3MaxScales       uint64
	E4M3InvalidScales   uint64
}

type NumericComparison struct {
	Role       string
	Name       string
	LeftDType  string
	RightDType string
	Values     uint64
	NonFinite  uint64
	SquaredErr float64
	LeftEnergy float64
	NMSE       float64
}

func shouldCompareNumerically(left, right *Tensor) bool {
	if left == nil || right == nil || left.CompanionOf != "" || right.CompanionOf != "" || !slices.Equal(logicalShape(left), logicalShape(right)) {
		return false
	}
	if !numericDType(left) || !numericDType(right) {
		return false
	}
	return payloadNeedsHash(left, right)
}

func numericDType(t *Tensor) bool {
	if t == nil {
		return false
	}
	switch t.ModelDType {
	case "BF16", "F16", "F32", "F64":
		return t.DType == t.ModelDType
	case "MXFP8", "MXFP4", "NVFP4":
		return t.Quantization != nil && t.DType == "U32"
	default:
		return false
	}
}

func payloadNeedsHash(left, right *Tensor) bool {
	if left == nil || right == nil {
		return true
	}
	if left.Blob != "" && left.Blob == right.Blob {
		return false
	}
	return left.File != right.File || left.Offset != right.Offset || left.Bytes != right.Bytes
}

func (h *payloadHasher) compareNumerically(ctx context.Context, left, right *Tensor, leftInv, rightInv *inventory) (*NumericComparison, error) {
	lr, err := newNumericReader(ctx, h, left, leftInv, rightInv, true)
	if err != nil {
		return nil, err
	}
	rr, err := newNumericReader(ctx, h, right, rightInv, leftInv, true)
	if err != nil {
		lr.abort()
		return nil, err
	}
	finished := false
	defer func() {
		if !finished {
			lr.abort()
			rr.abort()
		}
	}()

	const valuesPerChunk = 8192
	leftValues, rightValues := make([]float64, valuesPerChunk), make([]float64, valuesPerChunk)
	result := &NumericComparison{Role: left.Role, Name: left.Name, LeftDType: left.ModelDType, RightDType: right.ModelDType}
	for {
		ln, leftErr := lr.readValues(ctx, leftValues)
		rn, rightErr := rr.readValues(ctx, rightValues)
		if ln != rn {
			return nil, fmt.Errorf("decoded value counts differ: %d and %d", ln, rn)
		}
		for i := range ln {
			lv, rv := leftValues[i], rightValues[i]
			if math.IsNaN(lv) || math.IsNaN(rv) || math.IsInf(lv, 0) || math.IsInf(rv, 0) {
				result.NonFinite++
				continue
			}
			delta := rv - lv
			result.SquaredErr += delta * delta
			result.LeftEnergy += lv * lv
			result.Values++
		}
		if leftErr == io.EOF && rightErr == io.EOF {
			break
		}
		if leftErr != nil || rightErr != nil {
			return nil, fmt.Errorf("decode: left %v, right %v", leftErr, rightErr)
		}
	}
	if err := lr.finish(); err != nil {
		return nil, err
	}
	if err := rr.finish(); err != nil {
		return nil, err
	}
	finished = true
	if result.LeftEnergy == 0 {
		if result.SquaredErr == 0 {
			result.NMSE = 0
		} else {
			result.NMSE = math.Inf(1)
		}
	} else {
		result.NMSE = result.SquaredErr / result.LeftEnergy
	}
	h.stats.Left.add(lr.metrics)
	h.stats.Right.add(rr.metrics)
	h.statsScanned[left], h.statsScanned[right] = true, true
	return result, nil
}

func (h *payloadHasher) finishStats(ctx context.Context, changes []TensorChange, leftInv, rightInv *inventory) error {
	for _, change := range changes {
		left, right := change.Left, change.Right
		if left != nil && right != nil && quantizedForStats(left) && quantizedForStats(right) && !h.statsScanned[left] && !h.statsScanned[right] && !payloadNeedsHash(left, right) {
			metrics, err := h.scanQuantized(ctx, left, leftInv)
			if err != nil {
				return fmt.Errorf("tensor statistics %q: %w", tensorKey(left.Role, left.Name), err)
			}
			h.stats.Left.add(metrics)
			h.stats.Right.add(metrics)
			h.statsScanned[left], h.statsScanned[right] = true, true
			continue
		}
		for _, side := range []struct {
			tensor *Tensor
			inv    *inventory
			stats  *QuantStats
		}{{left, leftInv, &h.stats.Left}, {right, rightInv, &h.stats.Right}} {
			if side.tensor == nil || h.statsScanned[side.tensor] || !quantizedForStats(side.tensor) {
				continue
			}
			metrics, err := h.scanQuantized(ctx, side.tensor, side.inv)
			if err != nil {
				return fmt.Errorf("tensor statistics %q: %w", tensorKey(side.tensor.Role, side.tensor.Name), err)
			}
			side.stats.add(metrics)
			h.statsScanned[side.tensor] = true
		}
	}
	return nil
}

func quantizedForStats(t *Tensor) bool {
	return t != nil && t.CompanionOf == "" && slices.Contains([]string{"MXFP8", "MXFP4", "NVFP4"}, t.ModelDType) && t.Quantization != nil
}

func (h *payloadHasher) scanQuantized(ctx context.Context, tensor *Tensor, inv *inventory) (QuantStats, error) {
	reader, err := newNumericReader(ctx, h, tensor, inv, nil, false)
	if err != nil {
		return QuantStats{}, err
	}
	finished := false
	defer func() {
		if !finished {
			reader.abort()
		}
	}()
	buffer := make([]float64, 8192)
	for {
		_, err := reader.readValues(ctx, buffer)
		if err == io.EOF {
			break
		}
		if err != nil {
			return QuantStats{}, err
		}
	}
	if err := reader.finish(); err != nil {
		return QuantStats{}, err
	}
	finished = true
	return reader.metrics, nil
}

func (s *QuantStats) add(other QuantStats) {
	s.MXFP8Tensors += other.MXFP8Tensors
	s.MXFP4Tensors += other.MXFP4Tensors
	s.NVFP4Tensors += other.NVFP4Tensors
	s.E4M3PayloadBlocks += other.E4M3PayloadBlocks
	s.E4M3SaturatedBlocks += other.E4M3SaturatedBlocks
	s.E8M0Scales += other.E8M0Scales
	s.E8M0MaxScales += other.E8M0MaxScales
	s.E8M0InvalidScales += other.E8M0InvalidScales
	s.E4M3Scales += other.E4M3Scales
	s.E4M3MaxScales += other.E4M3MaxScales
	s.E4M3InvalidScales += other.E4M3InvalidScales
}

type numericReader struct {
	tensor    *Tensor
	payload   *statsPayloadReader
	kind      string
	remaining uint64
	scale     []byte
	scaleAt   int
	global    float64
	raw       []byte
	group     []float64
	groupAt   int
	metrics   QuantStats
}

func newNumericReader(ctx context.Context, h *payloadHasher, tensor *Tensor, inv, otherInv *inventory, hashPayloads bool) (*numericReader, error) {
	if !numericDType(tensor) {
		return nil, fmt.Errorf("unsupported numeric dtype %s stored as %s", tensor.ModelDType, tensor.DType)
	}
	reader := &numericReader{tensor: tensor, kind: tensor.ModelDType, global: 1}
	reader.remaining = tensor.Elements
	if tensor.Quantization != nil {
		reader.remaining = 1
		for _, dim := range tensor.Quantization.LogicalShape {
			if dim > 0 && reader.remaining > math.MaxUint64/dim {
				return nil, fmt.Errorf("logical element count overflows")
			}
			reader.remaining *= dim
		}
	}
	var err error
	reader.payload, err = newStatsPayloadReader(h, tensor, hashPayloads && payloadNeedsHash(tensor, matchingTensor(otherInv, tensor)))
	if err != nil {
		return nil, err
	}

	if tensor.Quantization == nil {
		return reader, nil
	}
	var scale, globalScale *Tensor
	for _, name := range tensor.Companions {
		companion := inv.tensors[tensorKey(tensor.Role, name)]
		switch {
		case strings.HasSuffix(name, ".global_scale"):
			globalScale = companion
		case strings.HasSuffix(name, ".scale"), strings.HasSuffix(name, ".scales"):
			scale = companion
		}
	}
	if scale == nil {
		reader.payload.abort()
		return nil, fmt.Errorf("%s has no scale companion", tensor.ModelDType)
	}
	otherScale := matchingTensor(otherInv, scale)
	reader.scale, err = readStatsTensor(ctx, h, scale, hashPayloads && payloadNeedsHash(scale, otherScale))
	if err != nil {
		reader.payload.abort()
		return nil, err
	}
	if globalScale != nil {
		data, err := readStatsTensor(ctx, h, globalScale, hashPayloads && payloadNeedsHash(globalScale, matchingTensor(otherInv, globalScale)))
		if err != nil {
			reader.payload.abort()
			return nil, err
		}
		if globalScale.DType != "F32" || len(data) != 4 {
			reader.payload.abort()
			return nil, fmt.Errorf("global scale is %s with %d bytes, want scalar F32", globalScale.DType, len(data))
		}
		reader.global = float64(math.Float32frombits(binary.LittleEndian.Uint32(data)))
	}
	switch tensor.ModelDType {
	case "MXFP8":
		reader.metrics.MXFP8Tensors = 1
	case "MXFP4":
		reader.metrics.MXFP4Tensors = 1
	case "NVFP4":
		reader.metrics.NVFP4Tensors = 1
	}
	reader.raw = make([]byte, tensor.Quantization.GroupSize*tensor.Quantization.Bits/8)
	reader.group = make([]float64, tensor.Quantization.GroupSize)
	reader.groupAt = len(reader.group)
	return reader, nil
}

func matchingTensor(inv *inventory, tensor *Tensor) *Tensor {
	if inv == nil || tensor == nil {
		return nil
	}
	return inv.tensors[tensorKey(tensor.Role, tensor.Name)]
}

func (r *numericReader) readValues(ctx context.Context, dst []float64) (int, error) {
	if r.remaining == 0 {
		return 0, io.EOF
	}
	if r.tensor.Quantization == nil {
		return r.readFloatValues(ctx, dst)
	}
	n := 0
	for n < len(dst) && r.remaining > 0 {
		if r.groupAt == len(r.group) {
			if err := r.decodeGroup(ctx); err != nil {
				return n, err
			}
		}
		count := min(len(dst)-n, len(r.group)-r.groupAt)
		if r.remaining < uint64(count) {
			count = int(r.remaining)
		}
		copy(dst[n:n+count], r.group[r.groupAt:r.groupAt+count])
		n += count
		r.groupAt += count
		r.remaining -= uint64(count)
	}
	return n, nil
}

func (r *numericReader) readFloatValues(ctx context.Context, dst []float64) (int, error) {
	size := map[string]int{"BF16": 2, "F16": 2, "F32": 4, "F64": 8}[r.kind]
	n := len(dst)
	if r.remaining < uint64(n) {
		n = int(r.remaining)
	}
	if len(r.raw) < len(dst)*size {
		r.raw = make([]byte, len(dst)*size)
	}
	raw := r.raw[:n*size]
	if err := r.payload.readFull(ctx, raw); err != nil {
		return 0, err
	}
	order := binary.LittleEndian
	var byteOrder binary.ByteOrder = order
	if r.tensor.ByteOrder == "big" {
		byteOrder = binary.BigEndian
	}
	for i := range n {
		at := i * size
		switch r.kind {
		case "BF16":
			dst[i] = float64(math.Float32frombits(uint32(byteOrder.Uint16(raw[at:])) << 16))
		case "F16":
			dst[i] = float64(decodeFloat16(byteOrder.Uint16(raw[at:])))
		case "F32":
			dst[i] = float64(math.Float32frombits(byteOrder.Uint32(raw[at:])))
		case "F64":
			dst[i] = math.Float64frombits(byteOrder.Uint64(raw[at:]))
		}
	}
	r.remaining -= uint64(n)
	return n, nil
}

func (r *numericReader) decodeGroup(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	bits := r.tensor.Quantization.Bits
	if err := r.payload.readFull(ctx, r.raw); err != nil {
		return err
	}
	if r.scaleAt >= len(r.scale) {
		return fmt.Errorf("scale data ended before weight groups")
	}
	scaleCode := r.scale[r.scaleAt]
	r.scaleAt++
	var scale float64
	switch r.kind {
	case "MXFP8", "MXFP4":
		r.metrics.E8M0Scales++
		if scaleCode == 0xfe {
			r.metrics.E8M0MaxScales++
		}
		if scaleCode == 0xff {
			r.metrics.E8M0InvalidScales++
		}
		scale = decodeE8M0(scaleCode)
	case "NVFP4":
		r.metrics.E4M3Scales++
		if scaleCode&0x7f == 0x7e {
			r.metrics.E4M3MaxScales++
		}
		if scaleCode&0x7f == 0x7f {
			r.metrics.E4M3InvalidScales++
		}
		scale = decodeE4M3(scaleCode)
	}
	scale *= r.global

	r.groupAt = 0
	if bits == 8 {
		saturated := false
		for i, code := range r.raw {
			if code&0x7f == 0x7e {
				saturated = true
			}
			r.group[i] = decodeE4M3(code) * scale
		}
		r.metrics.E4M3PayloadBlocks++
		if saturated {
			r.metrics.E4M3SaturatedBlocks++
		}
		return nil
	}
	for i, packed := range r.raw {
		r.group[2*i] = decodeE2M1(packed&0x0f) * scale
		r.group[2*i+1] = decodeE2M1(packed>>4) * scale
	}
	return nil
}

func (r *numericReader) finish() error {
	if r.remaining != 0 {
		return fmt.Errorf("%d decoded values remain", r.remaining)
	}
	if r.tensor.Quantization != nil && r.scaleAt != len(r.scale) {
		return fmt.Errorf("used %d of %d scale values", r.scaleAt, len(r.scale))
	}
	return r.payload.finish()
}

func (r *numericReader) abort() {
	if r != nil && r.payload != nil {
		r.payload.abort()
	}
}

type statsPayloadReader struct {
	h         *payloadHasher
	tensor    *Tensor
	reader    *io.SectionReader
	hash      hash.Hash
	hashBytes bool
	remaining int64
	file      *os.File
}

func newStatsPayloadReader(h *payloadHasher, tensor *Tensor, hashBytes bool) (*statsPayloadReader, error) {
	key := payloadRange{tensor.File, tensor.Offset, tensor.Bytes}
	if sum, ok := h.cache[key]; ok && hashBytes {
		tensor.SHA256 = sum
		hashBytes = false
	}
	file, err := os.Open(tensor.File)
	if err != nil {
		return nil, err
	}
	reader := &statsPayloadReader{h: h, tensor: tensor, reader: io.NewSectionReader(file, tensor.Offset, tensor.Bytes), hashBytes: hashBytes, remaining: tensor.Bytes, file: file}
	if hashBytes {
		reader.hash = sha256.New()
	}
	return reader, nil
}

func (r *statsPayloadReader) readFull(ctx context.Context, data []byte) error {
	if err := ctx.Err(); err != nil {
		r.file.Close()
		return err
	}
	if int64(len(data)) > r.remaining {
		r.file.Close()
		return io.ErrUnexpectedEOF
	}
	n, err := io.ReadFull(r.reader, data)
	if err != nil {
		r.file.Close()
		return err
	}
	r.remaining -= int64(n)
	r.h.stats.BytesRead += int64(n)
	if r.hashBytes {
		_, _ = r.hash.Write(data[:n])
		r.h.bytes += int64(n)
	} else {
		r.h.stats.ExtraBytesRead += int64(n)
	}
	return nil
}

func (r *statsPayloadReader) finish() error {
	defer r.file.Close()
	if r.remaining != 0 {
		return fmt.Errorf("%d unread payload bytes", r.remaining)
	}
	if r.hashBytes {
		sum := hex.EncodeToString(r.hash.Sum(nil))
		r.tensor.SHA256 = sum
		r.h.cache[payloadRange{r.tensor.File, r.tensor.Offset, r.tensor.Bytes}] = sum
	}
	return nil
}

func (r *statsPayloadReader) abort() {
	if r != nil && r.file != nil {
		_ = r.file.Close()
	}
}

func readStatsTensor(ctx context.Context, h *payloadHasher, tensor *Tensor, hashBytes bool) ([]byte, error) {
	if tensor.Bytes < 0 || uint64(tensor.Bytes) > uint64(^uint(0)>>1) {
		return nil, fmt.Errorf("tensor %q is too large to buffer", tensor.Name)
	}
	cacheKey := ""
	if !hashBytes && tensor.Blob != "" {
		cacheKey = fmt.Sprintf("%s@%d+%d", tensor.Blob, tensor.Offset, tensor.Bytes)
		if data, ok := h.statsData[cacheKey]; ok {
			return data, nil
		}
	}
	reader, err := newStatsPayloadReader(h, tensor, hashBytes)
	if err != nil {
		return nil, err
	}
	data := make([]byte, int(tensor.Bytes))
	if err := reader.readFull(ctx, data); err != nil {
		return nil, err
	}
	if err := reader.finish(); err != nil {
		return nil, err
	}
	if cacheKey != "" {
		h.statsData[cacheKey] = data
	}
	return data, nil
}

var e8m0Values = func() [256]float64 {
	var values [256]float64
	for i := range values {
		values[i] = decodeE8M0Slow(byte(i))
	}
	return values
}()

var e4m3Values = func() [256]float64 {
	var values [256]float64
	for i := range values {
		values[i] = decodeE4M3Slow(byte(i))
	}
	return values
}()

var e2m1Values = [...]float64{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}

func decodeE8M0(code byte) float64 { return e8m0Values[code] }

func decodeE8M0Slow(code byte) float64 {
	if code == 0xff {
		return math.NaN()
	}
	return math.Ldexp(1, int(code)-127)
}

func decodeE4M3(code byte) float64 { return e4m3Values[code] }

func decodeE4M3Slow(code byte) float64 {
	sign := 1.0
	if code&0x80 != 0 {
		sign = -1
	}
	exponent := int(code>>3) & 0x0f
	mantissa := int(code & 0x07)
	if exponent == 0x0f && mantissa == 0x07 {
		return math.NaN()
	}
	if exponent == 0 {
		return sign * math.Ldexp(float64(mantissa)/8, -6)
	}
	return sign * math.Ldexp(1+float64(mantissa)/8, exponent-7)
}

func decodeE2M1(code byte) float64 {
	return e2m1Values[code&0x0f]
}

func decodeFloat16(value uint16) float32 {
	sign := uint32(value&0x8000) << 16
	exponent := uint32(value>>10) & 0x1f
	mantissa := uint32(value & 0x03ff)
	switch exponent {
	case 0:
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		shift := 0
		for mantissa&0x0400 == 0 {
			mantissa <<= 1
			shift++
		}
		mantissa &= 0x03ff
		return math.Float32frombits(sign | uint32(127-15+1-shift)<<23 | mantissa<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mantissa<<13)
	default:
		return math.Float32frombits(sign | (exponent+127-15)<<23 | mantissa<<13)
	}
}
