package mlx

import (
	"fmt"
	"math"
	"sort"
	"testing"
)

func TestFastMoEExpertBlockMapMatchesExpertMap(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}

		const (
			tokens  = 73
			topK    = 4
			experts = 8
		)

		idxVals := make([]int32, tokens*topK)
		for tok := range tokens {
			idxVals[tok*topK] = int32(tok % experts)
			idxVals[tok*topK+1] = int32((tok + 1) % experts)
			idxVals[tok*topK+2] = int32((tok/2 + 3) % experts)
			idxVals[tok*topK+3] = int32((tok/3 + 5) % experts)
		}
		idx := FromValues(idxVals, tokens, topK)

		counts, ids, ok := fastMoEExpertMap(idx, experts)
		if !ok {
			t.Fatal("fastMoEExpertMap returned ok=false")
		}
		blockCounts, blockCount, blockExperts, blockOffsets, blockIDs, ok := fastMoEExpertBlockMap(idx, experts)
		if !ok {
			t.Fatal("fastMoEExpertBlockMap returned ok=false")
		}
		Eval(counts, ids, blockCounts, blockCount, blockExperts, blockOffsets, blockIDs)

		countVals := counts.Ints()
		blockCountVals := blockCounts.Ints()
		if len(countVals) != len(blockCountVals) {
			t.Fatalf("count lengths differ: %d vs %d", len(countVals), len(blockCountVals))
		}

		blockIDVals := blockIDs.Ints()
		expectedBlocks := 0
		expectedByExpert := make([][]int, experts)
		for flat, expert := range idxVals {
			expectedByExpert[int(expert)] = append(expectedByExpert[int(expert)], flat)
		}
		for expert, count := range countVals {
			if count != blockCountVals[expert] {
				t.Fatalf("expert %d count = %d, block count = %d", expert, count, blockCountVals[expert])
			}
			if count != len(expectedByExpert[expert]) {
				t.Fatalf("expert %d count = %d, want %d", expert, count, len(expectedByExpert[expert]))
			}
			expectedBlocks += (count + moeExpertBlockSize - 1) / moeExpertBlockSize
		}
		if got := blockCount.Ints()[0]; got != expectedBlocks {
			t.Fatalf("blockCount = %d, want %d", got, expectedBlocks)
		}

		seen := make(map[[2]int]bool)
		gotByExpert := make([][]int, experts)
		blockExpertVals := blockExperts.Ints()
		blockOffsetVals := blockOffsets.Ints()
		for i := range expectedBlocks {
			expert := blockExpertVals[i]
			offset := blockOffsetVals[i]
			if expert < 0 || expert >= experts {
				t.Fatalf("block %d expert = %d, want [0,%d)", i, expert, experts)
			}
			if offset < 0 || offset >= countVals[expert] || offset%moeExpertBlockSize != 0 {
				t.Fatalf("block %d offset = %d for expert %d count %d", i, offset, expert, countVals[expert])
			}
			key := [2]int{expert, offset}
			if seen[key] {
				t.Fatalf("duplicate block descriptor expert=%d offset=%d", expert, offset)
			}
			seen[key] = true

			limit := min(moeExpertBlockSize, countVals[expert]-offset)
			for j := range limit {
				flat := blockIDVals[i*moeExpertBlockSize+j]
				if flat < 0 || flat >= len(idxVals) {
					t.Fatalf("block %d id %d = %d, want [0,%d)", i, j, flat, len(idxVals))
				}
				if int(idxVals[flat]) != expert {
					t.Fatalf("block %d id %d routes to expert %d, want %d", i, j, idxVals[flat], expert)
				}
				gotByExpert[expert] = append(gotByExpert[expert], flat)
			}
		}
		for expert := range experts {
			for offset := 0; offset < countVals[expert]; offset += moeExpertBlockSize {
				if !seen[[2]int{expert, offset}] {
					t.Fatalf("missing block descriptor expert=%d offset=%d", expert, offset)
				}
			}
			sort.Ints(gotByExpert[expert])
			sort.Ints(expectedByExpert[expert])
			if len(gotByExpert[expert]) != len(expectedByExpert[expert]) {
				t.Fatalf("expert %d ids = %d, want %d", expert, len(gotByExpert[expert]), len(expectedByExpert[expert]))
			}
			for i := range gotByExpert[expert] {
				if gotByExpert[expert][i] != expectedByExpert[expert][i] {
					t.Fatalf("expert %d sorted id %d = %d, want %d", expert, i, gotByExpert[expert][i], expectedByExpert[expert][i])
				}
			}
		}
	})
}

func TestFastMoEExpertBlockMapRejectsLargeExpertSets(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}

		idx := FromValues([]int32{0, 1, 2, 3}, 1, 4)
		counts, blockCount, blockExperts, blockOffsets, blockIDs, ok := fastMoEExpertBlockMap(idx, maxMoEExpertBlockMapExperts+1)
		if ok {
			t.Fatal("fastMoEExpertBlockMap returned ok=true for an oversized expert set")
		}
		if counts != nil || blockCount != nil || blockExperts != nil || blockOffsets != nil || blockIDs != nil {
			t.Fatal("fastMoEExpertBlockMap returned outputs for an oversized expert set")
		}
	})
}

type fastMoEGatherQMMFunc func(x, w, scales, counts, ids *Array, topK int) (*Array, bool)

type fastMoEBlockGatherQMMFunc func(x, w, scales, counts, blockCount, blockExperts, blockOffsets, blockIDs *Array, topK int) (*Array, bool)

func TestFastNVFP4MoEGatherQMMMappedMatchesGatherQMM(t *testing.T) {
	testFastMoEGatherQMMMappedMatchesGatherQMM(t, "nvfp4", 16, 4,
		func(x, w, scales, counts, ids *Array, topK int) (*Array, bool) {
			return fastNVFP4MoEGatherQMMMapped(x, w, scales, counts, ids, topK, false)
		},
		func(x, w, scales, counts, blockCount, blockExperts, blockOffsets, blockIDs *Array, topK int) (*Array, bool) {
			return fastNVFP4MoEGatherQMMBlockMapped(x, w, scales, counts, blockCount, blockExperts, blockOffsets, blockIDs, topK, false)
		},
	)
}

func TestFastMXFP8MoEGatherQMMMappedMatchesGatherQMM(t *testing.T) {
	testFastMoEGatherQMMMappedMatchesGatherQMM(t, "mxfp8", 32, 8,
		func(x, w, scales, counts, ids *Array, topK int) (*Array, bool) {
			return fastMXFP8MoEGatherQMMMapped(x, w, scales, counts, ids, topK, false)
		},
		func(x, w, scales, counts, blockCount, blockExperts, blockOffsets, blockIDs *Array, topK int) (*Array, bool) {
			return fastMXFP8MoEGatherQMMBlockMapped(x, w, scales, counts, blockCount, blockExperts, blockOffsets, blockIDs, topK, false)
		},
	)
}

func TestFastMoEGatherMMBlockMappedMatchesGatherMM(t *testing.T) {
	var testErr error
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}
		failf := func(format string, args ...any) {
			if testErr == nil {
				testErr = fmt.Errorf(format, args...)
			}
		}
		dtype := DTypeBFloat16
		if !SupportsMoEGatherMMBlockMapped(dtype) {
			t.Skip("dense mapped MoE kernel unavailable")
		}

		for _, tc := range []struct {
			name    string
			tokens  int
			topK    int
			experts int
			inDim   int
			outDim  int
		}{
			{name: "small", tokens: 73, topK: 8, experts: 16, inDim: 128, outDim: 192},
			{name: "many_experts", tokens: 56, topK: 8, experts: 256, inDim: 128, outDim: 192},
			{name: "laguna_gate_up", tokens: 56, topK: 8, experts: 16, inDim: 2048, outDim: 1024},
			{name: "laguna_down", tokens: 56, topK: 8, experts: 16, inDim: 512, outDim: 2048},
		} {
			wVals := make([]float32, tc.experts*tc.inDim*tc.outDim)
			for i := range wVals {
				wVals[i] = float32((i%43)-21) * 0.005
			}
			w := FromValues(wVals, tc.experts, tc.inDim, tc.outDim).AsType(dtype)

			idxVals := make([]int32, tc.tokens*tc.topK)
			for tok := range tc.tokens {
				for k := range tc.topK {
					idxVals[tok*tc.topK+k] = int32((tok*5 + k*3 + k*k) % tc.experts)
				}
			}
			idx := FromValues(idxVals, tc.tokens, tc.topK)
			expertMap, ok := NewMoEExpertMap(idx, tc.experts)
			if !ok {
				failf("%s: NewMoEExpertMap returned ok=false", tc.name)
				return
			}

			if !runMoEGatherMMBlockMappedTestCase(tc.name+"_input_slots_1", tc.tokens, tc.topK, tc.inDim, tc.outDim, 1, dtype, w, w, idx, expertMap, failf) {
				return
			}
			if !runMoEGatherMMBlockMappedTestCase(tc.name+"_input_slots_topk", tc.tokens, tc.topK, tc.inDim, tc.outDim, tc.topK, dtype, w, w, idx, expertMap, failf) {
				return
			}

			sourceW := Transpose(w, 0, 2, 1).Clone()
			Eval(sourceW)
			if !runMoEGatherMMBlockMappedTestCase(tc.name+"_source_layout_input_slots_1", tc.tokens, tc.topK, tc.inDim, tc.outDim, 1, dtype, sourceW, w, idx, expertMap, failf) {
				return
			}
			if !runMoEGatherMMBlockMappedTestCase(tc.name+"_source_layout_input_slots_topk", tc.tokens, tc.topK, tc.inDim, tc.outDim, tc.topK, dtype, sourceW, w, idx, expertMap, failf) {
				return
			}
		}
	})
	if testErr != nil {
		t.Fatal(testErr)
	}
}

func runMoEGatherMMBlockMappedTestCase(name string, tokens, topK, inDim, outDim, inputSlots int, dtype DType, w, wantW, idx *Array, expertMap *MoEExpertMap, failf func(string, ...any)) bool {
	tokens32 := int32(tokens)
	topK32 := int32(topK)
	inDim32 := int32(inDim)
	outDim32 := int32(outDim)

	xVals := make([]float32, tokens*inputSlots*inDim)
	for i := range xVals {
		xVals[i] = float32((i%37)-18) * 0.01
	}
	x := FromValues(xVals, tokens, inputSlots, inDim).AsType(dtype)
	Pin(x, w, wantW, idx, expertMap.counts, expertMap.blockCount, expertMap.blockExperts, expertMap.blockOffsets, expertMap.blockIDs)
	defer Unpin(x, w, wantW, idx, expertMap.counts, expertMap.blockCount, expertMap.blockExperts, expertMap.blockOffsets, expertMap.blockIDs)

	got, ok := FastMoEGatherMMBlockMapped(x, w, expertMap, topK)
	if !ok {
		failf("%s: FastMoEGatherMMBlockMapped returned ok=false", name)
		return false
	}
	gotRelu, ok := FastMoEGatherMMBlockMappedReLUSquared(x, w, expertMap, topK)
	if !ok {
		failf("%s: FastMoEGatherMMBlockMappedReLUSquared returned ok=false", name)
		return false
	}

	zero := NewScalarArray(float32(0)).AsType(x.DType())
	xRelu := Maximum(x, zero)
	xRelu = Mul(xRelu, xRelu)

	var want *Array
	var wantRelu *Array
	if inputSlots == 1 {
		want = GatherMM(Reshape(x, tokens32, 1, 1, inDim32), wantW, nil, idx, false)
		wantRelu = GatherMM(Reshape(xRelu, tokens32, 1, 1, inDim32), wantW, nil, idx, false)
	} else {
		want = GatherMM(Reshape(x, tokens32, topK32, 1, inDim32), wantW, nil, idx, false)
		wantRelu = GatherMM(Reshape(xRelu, tokens32, topK32, 1, inDim32), wantW, nil, idx, false)
	}
	want = Reshape(want, tokens32, topK32, outDim32)
	wantRelu = Reshape(wantRelu, tokens32, topK32, outDim32)

	gotF := got.AsType(DTypeFloat32)
	gotReluF := gotRelu.AsType(DTypeFloat32)
	wantF := want.AsType(DTypeFloat32)
	wantReluF := wantRelu.AsType(DTypeFloat32)
	Eval(gotF, gotReluF, wantF, wantReluF)

	if dims := got.Dims(); len(dims) != 3 || dims[0] != tokens || dims[1] != topK || dims[2] != outDim {
		failf("%s: dims = %v, want [%d %d %d]", name, dims, tokens, topK, outDim)
		return false
	}
	if err := checkMoEFloat32Close(name+" dense block mapped", gotF.Floats(), wantF.Floats(), 3e-2); err != nil {
		failf("%v", err)
		return false
	}
	if err := checkMoEFloat32Close(name+" dense relu-squared block mapped", gotReluF.Floats(), wantReluF.Floats(), 3e-2); err != nil {
		failf("%v", err)
		return false
	}
	return true
}

func testFastMoEGatherQMMMappedMatchesGatherQMM(t *testing.T, mode string, groupSize, bits int, mapped fastMoEGatherQMMFunc, blockMapped fastMoEBlockGatherQMMFunc) {
	t.Helper()
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}

		const (
			tokens  = 73
			topK    = 8
			experts = 16
			inDim   = 128
			outDim  = 192
		)

		wVals := make([]float32, experts*outDim*inDim)
		for i := range wVals {
			wVals[i] = float32((i%41)-20) * 0.006
		}
		w := FromValues(wVals, experts, outDim, inDim).AsType(DTypeBFloat16)
		qw, scales, qbiases := Quantize(w, groupSize, bits, mode)
		if qbiases != nil {
			t.Fatalf("%s test expects no quantization bias", mode)
		}

		idxVals := make([]int32, tokens*topK)
		for tok := range tokens {
			for k := range topK {
				idxVals[tok*topK+k] = int32((tok*5 + k*3 + k*k) % experts)
			}
		}
		idx := FromValues(idxVals, tokens, topK)
		counts, ids, ok := fastMoEExpertMap(idx, experts)
		if !ok {
			t.Fatal("fastMoEExpertMap returned ok=false")
		}
		blockCounts, blockCount, blockExperts, blockOffsets, blockIDs, ok := fastMoEExpertBlockMap(idx, experts)
		if !ok {
			t.Fatal("fastMoEExpertBlockMap returned ok=false")
		}
		expertMap, ok := NewMoEGatherQMMMap(idx, experts)
		if !ok {
			t.Fatal("NewMoEGatherQMMMap returned ok=false")
		}

		run := func(name string, inputSlots int) {
			t.Helper()

			xVals := make([]float32, tokens*inputSlots*inDim)
			for i := range xVals {
				xVals[i] = float32((i%37)-18) * 0.01
			}
			x := FromValues(xVals, tokens, inputSlots, inDim).AsType(DTypeBFloat16)
			Pin(x, qw, scales, idx, counts, ids, blockCounts, blockIDs, blockCount, blockExperts, blockOffsets, expertMap.counts, expertMap.ids, expertMap.blockCount, expertMap.blockExperts, expertMap.blockOffsets, expertMap.blockIDs)
			defer Unpin(x, qw, scales, idx, counts, ids, blockCounts, blockIDs, blockCount, blockExperts, blockOffsets, expertMap.counts, expertMap.ids, expertMap.blockCount, expertMap.blockExperts, expertMap.blockOffsets, expertMap.blockIDs)

			got, ok := mapped(x, qw, scales, counts, ids, topK)
			if !ok {
				t.Fatalf("%s: %s mapped kernel returned ok=false", name, mode)
			}
			gotBlock, ok := blockMapped(x, qw, scales, blockCounts, blockCount, blockExperts, blockOffsets, blockIDs, topK)
			if !ok {
				t.Fatalf("%s: %s block mapped kernel returned ok=false", name, mode)
			}
			gotFast, ok := FastMoEGatherQMMBlockMapped(x, qw, scales, expertMap, topK, groupSize, bits, mode)
			if !ok {
				t.Fatalf("%s: %s generic mapped kernel returned ok=false", name, mode)
			}
			gotReluFast, ok := FastMoEGatherQMMBlockMappedReLUSquared(x, qw, scales, expertMap, topK, groupSize, bits, mode)
			if !ok {
				t.Fatalf("%s: %s generic relu-squared mapped kernel returned ok=false", name, mode)
			}

			zero := NewScalarArray(float32(0)).AsType(x.DType())
			xRelu := Maximum(x, zero)
			xRelu = Mul(xRelu, xRelu)

			var want *Array
			var wantRelu *Array
			if inputSlots == 1 {
				want = GatherQMM(Reshape(x, tokens, 1, 1, inDim), qw, scales, nil, nil, idx, true, groupSize, bits, mode, false)
				wantRelu = GatherQMM(Reshape(xRelu, tokens, 1, 1, inDim), qw, scales, nil, nil, idx, true, groupSize, bits, mode, false)
			} else {
				want = GatherQMM(Reshape(x, tokens, topK, 1, inDim), qw, scales, nil, nil, idx, true, groupSize, bits, mode, false)
				wantRelu = GatherQMM(Reshape(xRelu, tokens, topK, 1, inDim), qw, scales, nil, nil, idx, true, groupSize, bits, mode, false)
			}
			want = Reshape(want, tokens, topK, outDim)
			wantRelu = Reshape(wantRelu, tokens, topK, outDim)

			gotF := got.AsType(DTypeFloat32)
			gotBlockF := gotBlock.AsType(DTypeFloat32)
			gotFastF := gotFast.AsType(DTypeFloat32)
			gotReluFastF := gotReluFast.AsType(DTypeFloat32)
			wantF := want.AsType(DTypeFloat32)
			wantReluF := wantRelu.AsType(DTypeFloat32)
			Eval(gotF, gotBlockF, gotFastF, gotReluFastF, wantF, wantReluF)

			if dims := got.Dims(); len(dims) != 3 || dims[0] != tokens || dims[1] != topK || dims[2] != outDim {
				t.Fatalf("%s: dims = %v, want [%d %d %d]", name, dims, tokens, topK, outDim)
			}
			assertMoEFloat32Close(t, name+" mapped", gotF.Floats(), wantF.Floats(), 3e-2)
			assertMoEFloat32Close(t, name+" block mapped", gotBlockF.Floats(), wantF.Floats(), 3e-2)
			assertMoEFloat32Close(t, name+" generic mapped", gotFastF.Floats(), wantF.Floats(), 3e-2)
			assertMoEFloat32Close(t, name+" generic relu-squared mapped", gotReluFastF.Floats(), wantReluF.Floats(), 3e-2)
		}

		run("input_slots_1", 1)
		run("input_slots_topk", topK)
	})
}

func TestFastNVFP4MoEGatherQMMMappedMatchesGatherQMMWithCTPackedWeights(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}

		const (
			tokens  = 17
			topK    = 3
			experts = 4
			inDim   = 128
			outDim  = 64
		)

		words := make([]uint32, experts*outDim*(inDim/8))
		for e := range experts {
			for row := range outDim {
				rowWordBase := (e*outDim + row) * (inDim / 8)
				for word := range inDim / 8 {
					var packed uint32
					for byteInWord := range 4 {
						kByte := word*4 + byteInWord
						lo := uint8((e + row + kByte) & 7)
						hi := uint8((2*e + 3*row + kByte + 1) & 7)
						if (row+kByte)&3 == 0 {
							lo |= 8
						}
						if (e+kByte)&5 == 0 {
							hi |= 8
						}
						packed |= uint32(lo|(hi<<4)) << (8 * byteInWord)
					}
					words[rowWordBase+word] = packed
				}
			}
		}
		w := FromValues(words, experts, outDim, inDim/8)

		scaleVals := make([]uint8, experts*outDim*(inDim/16))
		scalePattern := []uint8{0x6f, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x7e}
		for i := range scaleVals {
			scaleVals[i] = scalePattern[i%len(scalePattern)]
		}
		scales := FromValues(scaleVals, experts, outDim, inDim/16)

		idxVals := make([]int32, tokens*topK)
		for tok := range tokens {
			for k := range topK {
				idxVals[tok*topK+k] = int32((tok + 2*k) % experts)
			}
		}
		idx := FromValues(idxVals, tokens, topK)
		counts, ids, ok := fastMoEExpertMap(idx, experts)
		if !ok {
			t.Fatal("fastMoEExpertMap returned ok=false")
		}

		run := func(name string, inputSlots int) {
			t.Helper()

			xVals := make([]float32, tokens*inputSlots*inDim)
			for i := range xVals {
				xVals[i] = float32((i%53)-26) * 0.003
			}
			x := FromValues(xVals, tokens, inputSlots, inDim).AsType(DTypeBFloat16)
			Pin(x, w, scales, idx, counts, ids)
			defer Unpin(x, w, scales, idx, counts, ids)

			got, ok := fastNVFP4MoEGatherQMMMapped(x, w, scales, counts, ids, topK, false)
			if !ok {
				t.Fatalf("%s: fastNVFP4MoEGatherQMMMapped returned ok=false", name)
			}

			var want *Array
			if inputSlots == 1 {
				want = GatherQMM(Reshape(x, tokens, 1, 1, inDim), w, scales, nil, nil, idx, true, 16, 4, "nvfp4", false)
			} else {
				want = GatherQMM(Reshape(x, tokens, topK, 1, inDim), w, scales, nil, nil, idx, true, 16, 4, "nvfp4", false)
			}
			want = Reshape(want, tokens, topK, outDim)

			gotF := got.AsType(DTypeFloat32)
			wantF := want.AsType(DTypeFloat32)
			Eval(gotF, wantF)

			assertMoEFloat32Close(t, name+" mapped", gotF.Floats(), wantF.Floats(), 3e-2)
		}

		run("input_slots_1", 1)
		run("input_slots_topk", topK)
	})
}

func assertMoEFloat32Close(t *testing.T, label string, got, want []float32, tol float64) {
	t.Helper()
	if err := checkMoEFloat32Close(label, got, want, tol); err != nil {
		t.Fatal(err)
	}
}

func checkMoEFloat32Close(label string, got, want []float32, tol float64) error {
	if len(got) != len(want) {
		return fmt.Errorf("%s: len got=%d want=%d", label, len(got), len(want))
	}
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			return fmt.Errorf("%s: got[%d]=%v want %v diff %v", label, i, got[i], want[i], diff)
		}
	}
	return nil
}
