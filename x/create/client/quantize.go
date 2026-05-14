package client

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/ollama/ollama/x/create"
	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/safetensors"
)

var clientQuantizePool = quantizeThreadPool{
	envKeys:      []string{"OLLAMA_CREATE_QUANTIZE_WORKERS"},
	defaultCount: 1,
}

type quantizeThreadPool struct {
	once         sync.Once
	threads      []*mlxthread.Thread
	err          error
	next         atomic.Uint32
	envKeys      []string
	defaultCount int
}

func (p *quantizeThreadPool) getThreads() ([]*mlxthread.Thread, error) {
	p.once.Do(func() {
		count := p.defaultCount
		for _, key := range p.envKeys {
			if s := strings.TrimSpace(os.Getenv(key)); s != "" {
				n, err := strconv.Atoi(s)
				if err == nil {
					count = n
				}
				break
			}
		}
		if count < 1 {
			count = 1
		}

		p.threads = make([]*mlxthread.Thread, 0, count)
		for i := range count {
			thread, err := mlxthread.Start(fmt.Sprintf("create-quantize-%d", i), func() error {
				if err := mlx.CheckInit(); err != nil {
					return err
				}
				mlx.BindCurrentThread()
				if mlx.GPUIsAvailable() {
					mlx.SetDefaultDeviceGPU()
				}
				return nil
			})
			if err != nil {
				p.err = err
				return
			}
			p.threads = append(p.threads, thread)
		}
	})

	return p.threads, p.err
}

func callOnQuantizePool[T any](p *quantizeThreadPool, fn func() (T, error)) (T, error) {
	threads, err := p.getThreads()
	if err != nil {
		var zero T
		return zero, err
	}

	thread := threads[int(p.next.Add(1)-1)%len(threads)]
	return mlxthread.Call(context.Background(), thread, fn)
}

func onQuantizeThread[T any](fn func() (T, error)) (T, error) {
	return callOnQuantizePool(&clientQuantizePool, fn)
}

func stageQuantizeInput(r io.Reader) (string, func(), error) {
	tmpFile, err := os.CreateTemp("", "ollama-quant-*.safetensors")
	if err != nil {
		return "", nil, fmt.Errorf("failed to create temp file: %w", err)
	}

	if _, err := io.Copy(tmpFile, r); err != nil {
		tmpPath := tmpFile.Name()
		tmpFile.Close()
		os.Remove(tmpPath)
		return "", nil, fmt.Errorf("failed to write temp file: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		tmpPath := tmpFile.Name()
		os.Remove(tmpPath)
		return "", nil, err
	}

	tmpPath := tmpFile.Name()
	return tmpPath, func() { os.Remove(tmpPath) }, nil
}

// loadAndQuantizeArrayPath loads a safetensors file with MLX, quantizes the tensor,
// and appends the resulting arrays (weight, scale, optional bias) to the provided
// map. If quantize is empty, the tensor is kept as-is.
func loadAndQuantizeArrayPath(inputPath, name, quantize string, arrays map[string]*mlx.Array) (toEval []*mlx.Array, nativeHandle *mlx.SafetensorsFile, err error) {
	if quantize != "" {
		if gs, _, _ := model.QuantizationParams(quantize); gs == 0 {
			return nil, nil, fmt.Errorf("unsupported quantization type: %s", quantize)
		}
	}

	st, err := mlx.LoadSafetensorsNative(inputPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load safetensors for %s: %w", name, err)
	}

	// Find the tensor key (may differ from name for single-tensor blobs)
	header, err := readSafetensorsHeader(inputPath)
	if err != nil {
		st.Free()
		return nil, nil, fmt.Errorf("failed to read blob header for %s: %w", name, err)
	}
	inputKey, err := safetensorsKey(name, header)
	if err != nil {
		st.Free()
		return nil, nil, fmt.Errorf("failed to resolve tensor key for %s: %w", name, err)
	}

	toEval, err = appendQuantizedArray(st, header, inputKey, name, quantize, arrays)
	if err != nil {
		st.Free()
		return nil, nil, err
	}
	return toEval, st, nil
}

func appendQuantizedArray(st *mlx.SafetensorsFile, header map[string]safetensorsHeaderEntry, inputKey, name, quantize string, arrays map[string]*mlx.Array) (toEval []*mlx.Array, err error) {
	return appendQuantizedArrayWithOptions(st, header, inputKey, name, quantize, arrays, true)
}

func appendQuantizedArrayWithOptions(st *mlx.SafetensorsFile, header map[string]safetensorsHeaderEntry, inputKey, name, quantize string, arrays map[string]*mlx.Array, eagerEval bool) (toEval []*mlx.Array, err error) {
	arr := st.Get(inputKey)
	if arr == nil {
		return nil, fmt.Errorf("tensor %q not found in safetensors", inputKey)
	}

	// Decode FP8 source encoding before checking quantize, so that callers
	// requesting decode-only (quantize="") receive usable float data.
	if info, ok := header[inputKey]; ok && info.Dtype == "F8_E4M3" {
		scaleKey := inputKey + ".scale_inv"
		scaleInv := st.Get(scaleKey)
		if scaleInv == nil {
			scaleKey = inputKey + ".scale"
			scaleInv = st.Get(scaleKey)
		}
		if scaleInv == nil {
			return nil, fmt.Errorf("missing companion tensor %q or %q for fp8 source tensor %q", inputKey+".scale_inv", inputKey+".scale", inputKey)
		}
		arr, err = decodeSourceFP8Tensor(arr, scaleInv)
		if err != nil {
			return nil, fmt.Errorf("failed to decode fp8 tensor %s: %w", inputKey, err)
		}
		if eagerEval {
			mlx.Eval(arr)
		}
	}

	if quantize == "" {
		arr = mlx.Contiguous(arr, false)
		arrays[name] = arr
		return []*mlx.Array{arr}, nil
	}

	if arr.DType() != mlx.DTypeBFloat16 && arr.DType() != mlx.DTypeFloat32 && arr.DType() != mlx.DTypeFloat16 {
		// Convert to float type if needed (quantize expects float)
		arr = arr.AsType(mlx.DTypeBFloat16)
		if eagerEval {
			mlx.Eval(arr)
		}
	}

	groupSize, bits, mode := model.QuantizationParams(quantize)
	qweight, scales, qbiases := mlx.Quantize(arr, groupSize, bits, mode)

	// Validate quantization produced non-empty output. MLX quantize may return
	// empty arrays for unsupported mode/bits combinations without raising an error.
	if eagerEval {
		mlx.Eval(qweight, scales)
	}
	if len(qweight.Dims()) == 0 || qweight.Dims()[0] == 0 {
		return nil, fmt.Errorf("mlx.Quantize produced empty weight for %s (quantize=%s, groupSize=%d, bits=%d, mode=%s)",
			name, quantize, groupSize, bits, mode)
	}
	if len(scales.Dims()) == 0 || scales.Dims()[0] == 0 {
		return nil, fmt.Errorf("mlx.Quantize produced empty scales for %s (quantize=%s, groupSize=%d, bits=%d, mode=%s)",
			name, quantize, groupSize, bits, mode)
	}

	qweight = mlx.Contiguous(qweight, false)
	scales = mlx.Contiguous(scales, false)
	arrays[name] = qweight
	arrays[name+".scale"] = scales
	toEval = append(toEval, qweight, scales)

	if qbiases != nil {
		qbiases = mlx.Contiguous(qbiases, false)
		arrays[name+".bias"] = qbiases
		toEval = append(toEval, qbiases)
	}

	return toEval, nil
}

func quantizeTensorPathWithRunner(runner func(func() (struct {
	path    string
	cleanup func()
}, error)) (struct {
	path    string
	cleanup func()
}, error), inputPath, tensorName, quantize string) (struct {
	path    string
	cleanup func()
}, error,
) {
	return runner(func() (struct {
		path    string
		cleanup func()
	}, error,
	) {
		arrays := make(map[string]*mlx.Array)
		toEval, st, err := loadAndQuantizeArrayPath(inputPath, tensorName, quantize, arrays)
		if err != nil {
			return struct {
				path    string
				cleanup func()
			}{}, err
		}

		finalArrays := make([]*mlx.Array, 0, len(arrays))
		for _, arr := range arrays {
			if arr != nil {
				finalArrays = append(finalArrays, arr)
			}
		}
		mlx.Pin(finalArrays...)
		defer func() {
			if st != nil {
				st.Free()
			}
			mlx.Unpin(finalArrays...)
			mlx.Sweep()
		}()

		mlx.Eval(toEval...)
		mlx.Sweep()
		// Free early to release mmap; defer guard handles error paths
		if st != nil {
			st.Free()
			st = nil
		}

		// Build metadata for single-tensor blobs
		groupSize, _, _ := model.QuantizationParams(quantize)
		metadata := map[string]string{
			"quant_type": quantize,
			"group_size": strconv.Itoa(groupSize),
		}

		outPath, cleanup, err := quantizeTempOutputPath("combined.safetensors")
		if err != nil {
			return struct {
				path    string
				cleanup func()
			}{}, err
		}
		if err := mlx.SaveSafetensorsWithMetadata(outPath, arrays, metadata); err != nil {
			cleanup()
			return struct {
				path    string
				cleanup func()
			}{}, fmt.Errorf("failed to save combined blob: %w", err)
		}
		return struct {
			path    string
			cleanup func()
		}{path: outPath, cleanup: cleanup}, nil
	})
}

// QuantizeTensorToFile loads a tensor from safetensors format, quantizes it,
// and writes a combined safetensors blob with the quantized weight, scale, and
// optional bias to a temp file. Tensor keys use the original tensor name:
// name, name.scale, name.bias.
func QuantizeTensorToFile(r io.Reader, tensorName, dtype string, shape []int32, quantize string) (string, func(), error) {
	inputPath, cleanupInput, err := stageQuantizeInput(r)
	if err != nil {
		return "", nil, err
	}
	defer cleanupInput()

	result, err := quantizeTensorPathWithRunner(onQuantizeThread, inputPath, tensorName, quantize)
	if err != nil {
		return "", nil, err
	}
	return result.path, result.cleanup, nil
}

// QuantizeTensorPath quantizes a safetensors blob that already exists on disk
// and writes the combined output safetensors blob to a temp file.
func QuantizeTensorPath(inputPath, tensorName, dtype string, shape []int32, quantize string) (string, func(), error) {
	result, err := quantizeTensorPathWithRunner(onQuantizeThread, inputPath, tensorName, quantize)
	if err != nil {
		return "", nil, err
	}
	return result.path, result.cleanup, nil
}

// QuantizeTensor loads a tensor from safetensors format, quantizes it,
// and returns a single combined safetensors blob with the quantized weight, scale, and optional bias.
func QuantizeTensor(r io.Reader, tensorName, dtype string, shape []int32, quantize string) (blobData []byte, err error) {
	outPath, cleanup, err := QuantizeTensorToFile(r, tensorName, dtype, shape, quantize)
	if err != nil {
		return nil, err
	}
	defer cleanup()
	return os.ReadFile(outPath)
}

// QuantizePackedGroup quantizes multiple tensors and saves them all into a single
// combined safetensors blob. Used for packing expert groups.
// When the inputs are per-expert 2D tensors (e.g., experts.0.gate_proj.weight),
// they are stacked into 3D switch_mlp tensors before quantization.
// Each tensor may have a different quantization type (mixed-precision).
func quantizePackedGroupToFileWithRunner(runner func(func() (struct {
	path    string
	cleanup func()
}, error)) (struct {
	path    string
	cleanup func()
}, error), groupName string, inputs []create.PackedTensorInput,
) (string, func(), error) {
	result, err := runner(func() (struct {
		path    string
		cleanup func()
	}, error,
	) {
		// Check if inputs are per-expert tensors that should be stacked into 3D
		if projGroups, projQuantize := parsePerExpertInputs(groupName, inputs); projGroups != nil {
			return stackAndQuantizeExpertGroup(groupName, projGroups, projQuantize)
		}

		stagedInputCleanup := func() {}
		stagedInputHeader := map[string]safetensorsHeaderEntry(nil)
		var stagedInputFile *mlx.SafetensorsFile
		defer func() {
			if stagedInputFile != nil {
				stagedInputFile.Free()
			}
			stagedInputCleanup()
		}()
		if allFileBackedPackedInputs(inputs) {
			stagedInputPath, cleanup, err := stagePackedTensorInputs(inputs, "packed-input.safetensors")
			if err != nil {
				return struct {
					path    string
					cleanup func()
				}{}, err
			}
			stagedInputCleanup = cleanup
			stagedInputHeader, err = readSafetensorsHeader(stagedInputPath)
			if err != nil {
				return struct {
					path    string
					cleanup func()
				}{}, fmt.Errorf("failed to read staged packed header: %w", err)
			}
			stagedInputFile, err = mlx.LoadSafetensorsNative(stagedInputPath)
			if err != nil {
				return struct {
					path    string
					cleanup func()
				}{}, fmt.Errorf("failed to load staged packed safetensors: %w", err)
			}
		}

		allArrays := make(map[string]*mlx.Array)
		var pinned []*mlx.Array

		var metadata map[string]string
		uniformQuantize := ""
		hasQuantized := false
		mixedQuantize := false
		for _, input := range inputs {
			if input.Quantize == "" {
				if hasQuantized {
					mixedQuantize = true
				}
				continue
			}
			if !hasQuantized {
				hasQuantized = true
				uniformQuantize = input.Quantize
				continue
			}
			if input.Quantize != uniformQuantize {
				mixedQuantize = true
			}
		}
		if hasQuantized && !mixedQuantize {
			if groupSize, _, _ := model.QuantizationParams(uniformQuantize); groupSize > 0 {
				metadata = map[string]string{
					"quant_type": uniformQuantize,
					"group_size": strconv.Itoa(groupSize),
				}
			}
		}

		for _, input := range inputs {
			var (
				toEval []*mlx.Array
				err    error
			)
			if stagedInputFile != nil {
				toEval, err = appendQuantizedArray(stagedInputFile, stagedInputHeader, input.Name, input.Name, input.Quantize, allArrays)
			} else {
				inputPath, cleanupInput, stageErr := stageQuantizeInput(input.Reader)
				if stageErr != nil {
					mlx.Unpin(pinned...)
					mlx.Sweep()
					return struct {
						path    string
						cleanup func()
					}{}, stageErr
				}

				toEval, _, err = loadAndQuantizeArrayPath(inputPath, input.Name, input.Quantize, allArrays)
				cleanupInput()
			}
			if err != nil {
				mlx.Unpin(pinned...)
				mlx.Sweep()
				return struct {
					path    string
					cleanup func()
				}{}, err
			}

			mlx.Eval(toEval...)

			finalArrays := arraysForPackedInput(allArrays, input)
			mlx.Pin(finalArrays...)
			pinned = append(pinned, finalArrays...)

			// Record per-tensor quant type so the model can resolve params at load time.
			if input.Quantize != "" {
				if groupSize, _, _ := model.QuantizationParams(input.Quantize); groupSize > 0 {
					if metadata == nil {
						metadata = make(map[string]string)
					}
					metadata[input.Name+".quant_type"] = input.Quantize
					metadata[input.Name+".group_size"] = strconv.Itoa(groupSize)
				}
			}
			mlx.Sweep()
		}
		defer func() {
			mlx.Unpin(pinned...)
			mlx.Sweep()
		}()

		// Save combined blob. Add global metadata only when every packed tensor uses
		// the same quantization mode and group size.
		outPath, cleanup, err := quantizeTempOutputPath("packed-combined.safetensors")
		if err != nil {
			return struct {
				path    string
				cleanup func()
			}{}, err
		}
		if err := mlx.SaveSafetensorsWithMetadata(outPath, allArrays, metadata); err != nil {
			cleanup()
			return struct {
				path    string
				cleanup func()
			}{}, fmt.Errorf("failed to save packed blob: %w", err)
		}

		return struct {
			path    string
			cleanup func()
		}{path: outPath, cleanup: cleanup}, nil
	})
	if err != nil {
		return "", nil, err
	}
	return result.path, result.cleanup, nil
}

func QuantizePackedGroupToFile(groupName string, inputs []create.PackedTensorInput) (string, func(), error) {
	return quantizePackedGroupToFileWithRunner(onQuantizeThread, groupName, inputs)
}

func QuantizePackedGroup(groupName string, inputs []create.PackedTensorInput) ([]byte, error) {
	outPath, cleanup, err := quantizePackedGroupToFileWithRunner(onQuantizeThread, groupName, inputs)
	if err != nil {
		return nil, err
	}
	defer cleanup()
	return os.ReadFile(outPath)
}

func arraysForPackedInput(allArrays map[string]*mlx.Array, input create.PackedTensorInput) []*mlx.Array {
	keys := []string{input.Name}
	if input.Quantize != "" {
		keys = append(keys, input.Name+".scale", input.Name+".bias")
	}

	out := make([]*mlx.Array, 0, len(keys))
	for _, key := range keys {
		if arr := allArrays[key]; arr != nil {
			out = append(out, arr)
		}
	}
	return out
}

func allFileBackedPackedInputs(inputs []create.PackedTensorInput) bool {
	for _, input := range inputs {
		if input.TensorData == nil {
			return false
		}
	}
	return len(inputs) > 0
}

// perExpertSuffix matches ".{index}.{proj_and_suffix}" after the group prefix.
var perExpertSuffix = regexp.MustCompile(`^\.(\d+)\.(.+)$`)

type expertTensorInfo struct {
	index int
	proj  string // e.g., "gate_proj.weight"
	input create.PackedTensorInput
}

// parsePerExpertInputs groups per-expert 2D tensor inputs by projection type
// and returns per-projection quantization types. Different projections may use
// different quant types (e.g., gate_up=int4, down=int8) but all experts within
// a projection must share the same type.
// Returns nil if the inputs are not per-expert tensors (e.g., already stacked 3D).
// Only handles ".experts" groups; ".shared_experts" groups are left unpacked.
func parsePerExpertInputs(groupName string, inputs []create.PackedTensorInput) (map[string][]expertTensorInfo, map[string]string) {
	if !strings.HasSuffix(groupName, ".experts") {
		return nil, nil
	}

	groups := make(map[string][]expertTensorInfo)
	projQuantize := make(map[string]string) // projection -> quant type
	for _, input := range inputs {
		suffix := strings.TrimPrefix(input.Name, groupName)
		m := perExpertSuffix.FindStringSubmatch(suffix)
		if m == nil {
			return nil, nil // not a per-expert pattern
		}
		index, err := strconv.Atoi(m[1])
		if err != nil {
			return nil, nil
		}
		proj := m[2]
		if existing, ok := projQuantize[proj]; ok {
			if input.Quantize != existing {
				return nil, nil // mixed quant within same projection
			}
		} else {
			projQuantize[proj] = input.Quantize
		}
		groups[proj] = append(groups[proj], expertTensorInfo{
			index: index,
			proj:  proj,
			input: input,
		})
	}
	if len(groups) == 0 {
		return nil, nil
	}
	return groups, projQuantize
}

func projectionNames(projGroups map[string][]expertTensorInfo) []string {
	projNames := make([]string, 0, len(projGroups))
	for proj := range projGroups {
		projNames = append(projNames, proj)
	}
	sort.Strings(projNames)
	return projNames
}

func fileBackedExpertInputs(projGroups map[string][]expertTensorInfo, projNames []string) ([]create.PackedTensorInput, bool) {
	inputs := make([]create.PackedTensorInput, 0)
	for _, proj := range projNames {
		experts := projGroups[proj]
		sort.Slice(experts, func(i, j int) bool {
			return experts[i].index < experts[j].index
		})
		for _, expert := range experts {
			if expert.input.TensorData == nil {
				return nil, false
			}
			inputs = append(inputs, expert.input)
		}
	}
	return inputs, true
}

func stagePackedTensorInputs(inputs []create.PackedTensorInput, name string) (string, func(), error) {
	tds := make([]*safetensors.TensorData, 0, len(inputs))
	for _, input := range inputs {
		if input.TensorData == nil {
			return "", nil, fmt.Errorf("packed tensor %s is not file-backed", input.Name)
		}
		tds = append(tds, input.TensorData.WithName(input.Name))
	}

	path, cleanup, err := quantizeTempOutputPath(name)
	if err != nil {
		return "", nil, err
	}

	f, err := os.Create(path)
	if err != nil {
		cleanup()
		return "", nil, err
	}
	if _, err := io.Copy(f, safetensors.BuildPackedSafetensorsReader(tds)); err != nil {
		f.Close()
		cleanup()
		return "", nil, fmt.Errorf("failed to stage packed inputs: %w", err)
	}
	if err := f.Close(); err != nil {
		cleanup()
		return "", nil, err
	}

	return path, cleanup, nil
}

// stackAndQuantizeExpertGroup decodes per-expert tensors, stacks them into 3D
// switch_mlp tensors, quantizes, and writes the combined safetensors blob to a temp file.
// projQuantize maps projection name to its quantization type (may differ per projection).
func stackAndQuantizeExpertGroup(groupName string, projGroups map[string][]expertTensorInfo, projQuantize map[string]string) (struct {
	path    string
	cleanup func()
}, error,
) {
	projNames := projectionNames(projGroups)
	if inputs, ok := fileBackedExpertInputs(projGroups, projNames); ok {
		return stackAndQuantizeExpertGroupFast(groupName, projGroups, projQuantize, projNames, inputs)
	}
	return stackAndQuantizeExpertGroupSlow(groupName, projGroups, projQuantize, projNames)
}

func stackAndQuantizeExpertGroupFast(groupName string, projGroups map[string][]expertTensorInfo, projQuantize map[string]string, projNames []string, inputs []create.PackedTensorInput) (struct {
	path    string
	cleanup func()
}, error,
) {
	inputPath, cleanupInput, err := stagePackedTensorInputs(inputs, "stacked-input.safetensors")
	if err != nil {
		return struct {
			path    string
			cleanup func()
		}{}, err
	}
	defer cleanupInput()

	header, err := readSafetensorsHeader(inputPath)
	if err != nil {
		return struct {
			path    string
			cleanup func()
		}{}, fmt.Errorf("failed to read staged packed header: %w", err)
	}

	st, err := mlx.LoadSafetensorsNative(inputPath)
	if err != nil {
		return struct {
			path    string
			cleanup func()
		}{}, fmt.Errorf("failed to load staged packed safetensors: %w", err)
	}
	defer st.Free()

	groupBase := strings.TrimSuffix(groupName, ".experts")
	allArrays := make(map[string]*mlx.Array)
	metadata := make(map[string]string)
	finalArrays := make([]*mlx.Array, 0, len(inputs)*2)

	type quantizedProjection struct {
		name      string
		quantize  string
		groupSize int
		bits      int
		mode      string
		weight    *mlx.Array
		scales    *mlx.Array
	}
	validations := make([]quantizedProjection, 0, len(projNames))

	for _, proj := range projNames {
		experts := projGroups[proj]
		sort.Slice(experts, func(i, j int) bool {
			return experts[i].index < experts[j].index
		})

		decoded := make([]*mlx.Array, 0, len(experts))
		for _, expert := range experts {
			arr := st.Get(expert.input.Name)
			if arr == nil {
				return struct {
					path    string
					cleanup func()
				}{}, fmt.Errorf("tensor %q not found in staged packed safetensors", expert.input.Name)
			}
			if info, ok := header[expert.input.Name]; ok && info.Dtype == "F8_E4M3" {
				scaleKey := expert.input.Name + ".scale_inv"
				scaleInv := st.Get(scaleKey)
				if scaleInv == nil {
					scaleKey = expert.input.Name + ".scale"
					scaleInv = st.Get(scaleKey)
				}
				if scaleInv == nil {
					return struct {
						path    string
						cleanup func()
					}{}, fmt.Errorf("missing companion tensor %q or %q for fp8 source tensor %q", expert.input.Name+".scale_inv", expert.input.Name+".scale", expert.input.Name)
				}
				arr, err = decodeSourceFP8Tensor(arr, scaleInv)
				if err != nil {
					return struct {
						path    string
						cleanup func()
					}{}, fmt.Errorf("failed to decode fp8 expert tensor %s: %w", expert.input.Name, err)
				}
			}
			decoded = append(decoded, arr)
		}

		stacked := mlx.Stack(decoded, 0)
		stackedName := groupBase + ".switch_mlp." + proj
		quantize := projQuantize[proj]

		if quantize != "" {
			groupSize, bits, mode := model.QuantizationParams(quantize)
			if groupSize > 0 {
				metadata[stackedName+".quant_type"] = quantize
				metadata[stackedName+".group_size"] = strconv.Itoa(groupSize)
			}

			qweight, scales, qbiases := mlx.Quantize(stacked, groupSize, bits, mode)
			qweight = mlx.Contiguous(qweight, false)
			scales = mlx.Contiguous(scales, false)
			allArrays[stackedName] = qweight
			allArrays[stackedName+".scale"] = scales
			finalArrays = append(finalArrays, qweight, scales)
			validations = append(validations, quantizedProjection{
				name:      stackedName,
				quantize:  quantize,
				groupSize: groupSize,
				bits:      bits,
				mode:      mode,
				weight:    qweight,
				scales:    scales,
			})

			if qbiases != nil {
				qbiases = mlx.Contiguous(qbiases, false)
				allArrays[stackedName+".bias"] = qbiases
				finalArrays = append(finalArrays, qbiases)
			}
			continue
		}

		stacked = mlx.Contiguous(stacked, false)
		allArrays[stackedName] = stacked
		finalArrays = append(finalArrays, stacked)
	}

	mlx.Eval(finalArrays...)
	for _, v := range validations {
		if len(v.weight.Dims()) == 0 || v.weight.Dims()[0] == 0 {
			return struct {
					path    string
					cleanup func()
				}{}, fmt.Errorf("mlx.Quantize produced empty weight for %s (quantize=%s, groupSize=%d, bits=%d, mode=%s)",
					v.name, v.quantize, v.groupSize, v.bits, v.mode)
		}
		if len(v.scales.Dims()) == 0 || v.scales.Dims()[0] == 0 {
			return struct {
					path    string
					cleanup func()
				}{}, fmt.Errorf("mlx.Quantize produced empty scales for %s (quantize=%s, groupSize=%d, bits=%d, mode=%s)",
					v.name, v.quantize, v.groupSize, v.bits, v.mode)
		}
	}

	mlx.Pin(finalArrays...)
	defer func() {
		mlx.Unpin(finalArrays...)
		mlx.Sweep()
	}()
	mlx.Sweep()

	outPath, cleanupOutput, err := quantizeTempOutputPath("stacked-combined.safetensors")
	if err != nil {
		return struct {
			path    string
			cleanup func()
		}{}, err
	}
	if err := mlx.SaveSafetensorsWithMetadata(outPath, allArrays, metadata); err != nil {
		cleanupOutput()
		return struct {
			path    string
			cleanup func()
		}{}, fmt.Errorf("failed to save stacked blob: %w", err)
	}
	return struct {
		path    string
		cleanup func()
	}{path: outPath, cleanup: cleanupOutput}, nil
}

func stackAndQuantizeExpertGroupSlow(groupName string, projGroups map[string][]expertTensorInfo, projQuantize map[string]string, projNames []string) (struct {
	path    string
	cleanup func()
}, error,
) {
	groupBase := strings.TrimSuffix(groupName, ".experts")

	allArrays := make(map[string]*mlx.Array)
	var pinned []*mlx.Array

	// Build metadata: if all projections use the same quant type, set global metadata.
	// Otherwise record per-tensor quant info.
	metadata := make(map[string]string)

	cleanup := func() {
		for _, p := range pinned {
			if p != nil {
				mlx.Unpin(p)
			}
		}
		mlx.Sweep()
	}

	for _, proj := range projNames {
		experts := projGroups[proj]

		// Sort by expert index
		sort.Slice(experts, func(i, j int) bool {
			return experts[i].index < experts[j].index
		})

		// Load and decode each expert tensor
		var decoded []*mlx.Array
		for _, expert := range experts {
			dummyArrays := make(map[string]*mlx.Array)
			inputPath, cleanupInput, err := stageQuantizeInput(expert.input.Reader)
			if err != nil {
				cleanup()
				return struct {
					path    string
					cleanup func()
				}{}, err
			}
			toEval, st, err := loadAndQuantizeArrayPath(inputPath, expert.input.Name, "", dummyArrays)
			cleanupInput()
			if err != nil {
				cleanup()
				return struct {
					path    string
					cleanup func()
				}{}, fmt.Errorf("failed to decode expert tensor %s: %w", expert.input.Name, err)
			}
			mlx.Eval(toEval...)

			arr := dummyArrays[expert.input.Name]
			mlx.Pin(arr)
			pinned = append(pinned, arr)
			decoded = append(decoded, arr)

			if st != nil {
				st.Free()
			}
			mlx.Sweep()
		}

		// Stack into 3D along axis 0: [numExperts, rows, cols]
		stacked := mlx.Stack(decoded, 0)
		mlx.Eval(stacked)
		mlx.Pin(stacked)
		pinned = append(pinned, stacked)

		// Free individual decoded arrays (remove from pinned to avoid double-unpin in cleanup)
		for i, p := range pinned {
			for _, d := range decoded {
				if p == d {
					pinned[i] = nil
				}
			}
		}
		mlx.Unpin(decoded...)
		mlx.Sweep()

		stackedName := groupBase + ".switch_mlp." + proj
		quantize := projQuantize[proj]

		// Record per-tensor quant metadata so the model can resolve params at load time.
		if quantize != "" {
			if groupSize, _, _ := model.QuantizationParams(quantize); groupSize > 0 {
				metadata[stackedName+".quant_type"] = quantize
				metadata[stackedName+".group_size"] = strconv.Itoa(groupSize)
			}
		}

		// Quantize the stacked tensor
		if quantize != "" {
			groupSize, bits, mode := model.QuantizationParams(quantize)

			qweight, scales, qbiases := mlx.Quantize(stacked, groupSize, bits, mode)

			// Validate quantization produced non-empty output.
			mlx.Eval(qweight, scales)
			if len(qweight.Dims()) == 0 || qweight.Dims()[0] == 0 {
				cleanup()
				return struct {
						path    string
						cleanup func()
					}{}, fmt.Errorf("mlx.Quantize produced empty weight for %s (quantize=%s, groupSize=%d, bits=%d, mode=%s)",
						stackedName, quantize, groupSize, bits, mode)
			}

			qweight = mlx.Contiguous(qweight, false)
			scales = mlx.Contiguous(scales, false)
			allArrays[stackedName] = qweight
			allArrays[stackedName+".scale"] = scales

			toEval := []*mlx.Array{qweight, scales}
			if qbiases != nil {
				qbiases = mlx.Contiguous(qbiases, false)
				allArrays[stackedName+".bias"] = qbiases
				toEval = append(toEval, qbiases)
			}
			mlx.Eval(toEval...)
			mlx.Pin(toEval...)
			pinned = append(pinned, toEval...)

			// Free stacked source array (remove from pinned to avoid double-unpin in cleanup)
			for i, p := range pinned {
				if p == stacked {
					pinned[i] = nil
				}
			}
			mlx.Unpin(stacked)
			mlx.Sweep()
		} else {
			stacked = mlx.Contiguous(stacked, false)
			mlx.Eval(stacked)
			mlx.Pin(stacked)
			pinned = append(pinned, stacked)
			allArrays[stackedName] = stacked
		}
	}

	defer cleanup()

	outPath, cleanupOutput, err := quantizeTempOutputPath("stacked-combined.safetensors")
	if err != nil {
		return struct {
			path    string
			cleanup func()
		}{}, err
	}
	if err := mlx.SaveSafetensorsWithMetadata(outPath, allArrays, metadata); err != nil {
		cleanupOutput()
		return struct {
			path    string
			cleanup func()
		}{}, fmt.Errorf("failed to save stacked blob: %w", err)
	}
	return struct {
		path    string
		cleanup func()
	}{path: outPath, cleanup: cleanupOutput}, nil
}

// QuantizeSupported returns true if quantization is supported (MLX library available)
func QuantizeSupported() bool {
	return mlx.CheckInit() == nil
}

func quantizeTempOutputPath(name string) (string, func(), error) {
	tmpDir, err := os.MkdirTemp("", "ollama-quantize-*")
	if err != nil {
		return "", nil, fmt.Errorf("failed to create quantization temp dir: %w", err)
	}
	return filepath.Join(tmpDir, name), func() { os.RemoveAll(tmpDir) }, nil
}

type safetensorsHeaderEntry struct {
	Dtype string  `json:"dtype"`
	Shape []int32 `json:"shape"`
}

func readSafetensorsHeader(path string) (map[string]safetensorsHeaderEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return nil, err
	}
	if headerSize > safetensors.MaxHeaderSize {
		return nil, fmt.Errorf("safetensors header too large: %d bytes", headerSize)
	}
	if stat, err := f.Stat(); err != nil {
		return nil, err
	} else if stat.Size() < 8 {
		return nil, fmt.Errorf("safetensors file is too small: %d bytes", stat.Size())
	} else if headerSize > uint64(stat.Size()-8) {
		return nil, fmt.Errorf("safetensors header size %d exceeds file size %d", headerSize, stat.Size())
	}
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return nil, err
	}

	var header map[string]safetensorsHeaderEntry
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, err
	}
	return header, nil
}

// safetensorsKey resolves the primary tensor key from a header.
func safetensorsKey(preferred string, header map[string]safetensorsHeaderEntry) (string, error) {
	if preferred != "" {
		if _, ok := header[preferred]; ok {
			return preferred, nil
		}
	}

	keys := make([]string, 0, len(header))
	for k := range header {
		if k == "__metadata__" || strings.HasSuffix(k, ".scale_inv") {
			continue
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)
	if len(keys) == 0 {
		return "", fmt.Errorf("no tensor found in safetensors header")
	}
	return keys[0], nil
}

func decodeSourceFP8Tensor(weight, scale *mlx.Array) (*mlx.Array, error) {
	if weight == nil || scale == nil {
		return nil, fmt.Errorf("fp8 weight and scale tensors are required")
	}

	weightShape := weight.Dims()
	scaleShape := scale.Dims()
	if len(weightShape) != 2 || len(scaleShape) != 2 {
		return nil, fmt.Errorf("expected 2D fp8 weight and scale tensors, got %v and %v", weightShape, scaleShape)
	}

	// These must match the block size validated by resolveEffectiveQuantization
	// in create.go, which rejects any source model with a different block size.
	const blockRows = 128
	const blockCols = 128
	rows, cols := weightShape[0], weightShape[1]
	expectedScaleRows := (rows + blockRows - 1) / blockRows
	expectedScaleCols := (cols + blockCols - 1) / blockCols
	if scaleShape[0] != expectedScaleRows || scaleShape[1] != expectedScaleCols {
		return nil, fmt.Errorf(
			"unexpected fp8 scale shape %v for weight shape %v; want [%d %d]",
			scaleShape,
			weightShape,
			expectedScaleRows,
			expectedScaleCols,
		)
	}

	decoded := mlx.FromFP8(weight, mlx.DTypeBFloat16)
	padBottom := blockRows*scaleShape[0] - rows
	padSide := blockCols*scaleShape[1] - cols
	if padBottom > 0 || padSide > 0 {
		decoded = mlx.PadConstant(decoded, []int{0, 1}, []int{0, 0}, []int{padBottom, padSide})
	}

	decoded = mlx.Reshape(decoded, int32(scaleShape[0]), int32(blockRows), int32(scaleShape[1]), int32(blockCols))
	decoded = mlx.Mul(decoded, mlx.ExpandDims(mlx.ExpandDims(scale, 1), 3))
	decoded = mlx.Reshape(decoded, int32(rows+padBottom), int32(cols+padSide))
	if padBottom > 0 || padSide > 0 {
		decoded = mlx.SliceStartStop(decoded, []int32{0, 0}, []int32{int32(rows), int32(cols)})
	}

	return decoded, nil
}
