package llm

import (
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// QuickCheck-style generators for comprehensive property testing
// Based on Haskell QuickCheck and similar property-based testing frameworks

// Generator interface for creating random test data
type Generator[T any] interface {
	Generate(r *rand.Rand) T
	Shrink(value T) []T
}

// LayerCountGenerator generates realistic transformer layer counts
type LayerCountGenerator struct {
	MinLayers int
	MaxLayers int
}

func NewLayerCountGenerator() *LayerCountGenerator {
	return &LayerCountGenerator{
		MinLayers: 1,
		MaxLayers: 200, // Realistic range for transformers
	}
}

func (g *LayerCountGenerator) Generate(r *rand.Rand) int {
	// Bias towards common sizes: 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128
	commonSizes := []int{4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128}

	if r.Float32() < 0.7 { // 70% chance of common size
		return commonSizes[r.Intn(len(commonSizes))]
	}

	// 30% chance of random size in range
	return r.Intn(g.MaxLayers-g.MinLayers+1) + g.MinLayers
}

func (g *LayerCountGenerator) Shrink(value int) []int {
	if value <= g.MinLayers {
		return nil
	}

	shrunk := make([]int, 0, 5)

	// Try smaller common sizes
	candidates := []int{1, 4, 8, 16, 32, 64}
	for _, candidate := range candidates {
		if candidate < value && candidate >= g.MinLayers {
			shrunk = append(shrunk, candidate)
		}
	}

	// Try value - 1
	if value-1 >= g.MinLayers {
		shrunk = append(shrunk, value-1)
	}

	// Try value / 2
	if value/2 >= g.MinLayers {
		shrunk = append(shrunk, value/2)
	}

	return shrunk
}

// KVCacheSizeGenerator generates realistic KV cache sizes
type KVCacheSizeGenerator struct {
	MinSize uint64
	MaxSize uint64
}

func NewKVCacheSizeGenerator() *KVCacheSizeGenerator {
	return &KVCacheSizeGenerator{
		MinSize: 1,
		MaxSize: 100000,
	}
}

func (g *KVCacheSizeGenerator) Generate(r *rand.Rand) uint64 {
	// Bias towards multiples of 28 (MLA tile size)
	if r.Float32() < 0.5 {
		multiplier := r.Intn(1000) + 1
		return uint64(multiplier * 28)
	}

	// Bias towards powers of 2
	if r.Float32() < 0.3 {
		power := r.Intn(16) + 1 // 2^1 to 2^16
		size := uint64(1 << power)
		if size <= g.MaxSize {
			return size
		}
	}

	// Random size in range
	return uint64(r.Intn(int(g.MaxSize-g.MinSize+1))) + g.MinSize
}

func (g *KVCacheSizeGenerator) Shrink(value uint64) []uint64 {
	if value <= g.MinSize {
		return nil
	}

	shrunk := make([]uint64, 0, 10)

	// Try common sizes
	candidates := []uint64{1, 28, 56, 280, 560, 1000, 2800}
	for _, candidate := range candidates {
		if candidate < value && candidate >= g.MinSize {
			shrunk = append(shrunk, candidate)
		}
	}

	// Try value - 1
	if value-1 >= g.MinSize {
		shrunk = append(shrunk, value-1)
	}

	// Try value / 2
	if value/2 >= g.MinSize {
		shrunk = append(shrunk, value/2)
	}

	// Try nearest multiple of 28
	if value > 28 {
		nearest28 := (value / 28) * 28
		if nearest28 >= g.MinSize && nearest28 < value {
			shrunk = append(shrunk, nearest28)
		}
	}

	return shrunk
}

// GPUSpecGenerator generates realistic GPU specifications
type GPUSpecGenerator struct{}

func NewGPUSpecGenerator() *GPUSpecGenerator {
	return &GPUSpecGenerator{}
}

type GPUTemplate struct {
	Vendor              GPUVendor
	MemoryRange         [2]int    // [min, max] GB
	TFLOPSRange         [2]int    // [min, max] TFLOPS
	ComputeUnitsRange   [2]int    // [min, max] units
	TensorCoreProb      float32   // Probability of having tensor cores
	BandwidthRange      [2]int    // [min, max] GB/s
	ArchitectureOptions []GPUArchitecture
	NameTemplates       []string
}

func (g *GPUSpecGenerator) getTemplates() []GPUTemplate {
	return []GPUTemplate{
		{
			Vendor:              NVIDIA,
			MemoryRange:         [2]int{4, 48},
			TFLOPSRange:         [2]int{10, 80},
			ComputeUnitsRange:   [2]int{16, 128},
			TensorCoreProb:      0.8,
			BandwidthRange:      [2]int{300, 1000},
			ArchitectureOptions: []GPUArchitecture{AMPERE},
			NameTemplates:       []string{"RTX %d0", "GTX %d0", "Tesla %c%d00"},
		},
		{
			Vendor:              AMD,
			MemoryRange:         [2]int{4, 32},
			TFLOPSRange:         [2]int{8, 60},
			ComputeUnitsRange:   [2]int{20, 80},
			TensorCoreProb:      0.1,
			BandwidthRange:      [2]int{400, 800},
			ArchitectureOptions: []GPUArchitecture{RDNA2},
			NameTemplates:       []string{"RX %d00 XT", "Radeon %d00"},
		},
		{
			Vendor:              INTEL,
			MemoryRange:         [2]int{6, 16},
			TFLOPSRange:         [2]int{10, 25},
			ComputeUnitsRange:   [2]int{16, 32},
			TensorCoreProb:      0.6,
			BandwidthRange:      [2]int{300, 500},
			ArchitectureOptions: []GPUArchitecture{XE_HPG},
			NameTemplates:       []string{"Arc A%d0", "Arc B%d0"},
		},
		{
			Vendor:              APPLE,
			MemoryRange:         [2]int{8, 64},
			TFLOPSRange:         [2]int{5, 30},
			ComputeUnitsRange:   [2]int{8, 40},
			TensorCoreProb:      0.0,
			BandwidthRange:      [2]int{200, 400},
			ArchitectureOptions: []GPUArchitecture{M_SERIES},
			NameTemplates:       []string{"M%d Pro", "M%d Max", "M%d Ultra"},
		},
	}
}

func (g *GPUSpecGenerator) Generate(r *rand.Rand) *GPUDeviceSpec {
	templates := g.getTemplates()
	template := templates[r.Intn(len(templates))]

	// Generate values within template ranges
	memory := r.Intn(template.MemoryRange[1]-template.MemoryRange[0]+1) + template.MemoryRange[0]
	tflops := r.Intn(template.TFLOPSRange[1]-template.TFLOPSRange[0]+1) + template.TFLOPSRange[0]
	computeUnits := r.Intn(template.ComputeUnitsRange[1]-template.ComputeUnitsRange[0]+1) + template.ComputeUnitsRange[0]
	tensorCores := r.Float32() < template.TensorCoreProb
	bandwidth := r.Intn(template.BandwidthRange[1]-template.BandwidthRange[0]+1) + template.BandwidthRange[0]

	// Generate name
	nameTemplate := template.NameTemplates[r.Intn(len(template.NameTemplates))]
	var name string
	if strings.Contains(nameTemplate, "%d") {
		name = fmt.Sprintf(nameTemplate, r.Intn(9)+1)
	} else if strings.Contains(nameTemplate, "%c") {
		letter := 'A' + rune(r.Intn(26))
		number := r.Intn(9) + 1
		name = fmt.Sprintf(nameTemplate, letter, number)
	} else {
		name = nameTemplate
	}

	return &GPUDeviceSpec{
		Vendor:              template.Vendor,
		MemorySizeGB:        memory,
		ComputeUnits:        computeUnits,
		PeakTFLOPSFP32:      tflops,
		PeakTFLOPSFP16:      tflops * 2, // Rough approximation
		SupportsTensorCores: tensorCores,
		SupportsHalfPrec:    true,
		MemoryBandwidthGBps: bandwidth,
		DeviceName:          name,
	}
}

func (g *GPUSpecGenerator) Shrink(value *GPUDeviceSpec) []*GPUDeviceSpec {
	shrunk := make([]*GPUDeviceSpec, 0, 5)

	// Shrink memory
	if value.MemorySizeGB > 4 {
		smaller := *value
		smaller.MemorySizeGB = value.MemorySizeGB / 2
		if smaller.MemorySizeGB < 4 {
			smaller.MemorySizeGB = 4
		}
		shrunk = append(shrunk, &smaller)
	}

	// Remove tensor cores
	if value.SupportsTensorCores {
		smaller := *value
		smaller.SupportsTensorCores = false
		shrunk = append(shrunk, &smaller)
	}

	// Shrink TFLOPS
	if value.PeakTFLOPSFP32 > 5 {
		smaller := *value
		smaller.PeakTFLOPSFP32 = value.PeakTFLOPSFP32 / 2
		smaller.PeakTFLOPSFP16 = smaller.PeakTFLOPSFP32 * 2
		shrunk = append(shrunk, &smaller)
	}

	return shrunk
}

// WorkloadGenerator creates realistic ML workloads
type WorkloadGenerator struct{}

func NewWorkloadGenerator() *WorkloadGenerator {
	return &WorkloadGenerator{}
}

func (g *WorkloadGenerator) Generate(r *rand.Rand) []Operation {
	numOps := r.Intn(10) + 1 // 1-10 operations
	ops := make([]Operation, numOps)

	for i := 0; i < numOps; i++ {
		ops[i] = g.generateSingleOperation(r)
	}

	return ops
}

func (g *WorkloadGenerator) generateSingleOperation(r *rand.Rand) Operation {
	opTypes := []OperationType{UNIFIED_GEMM, SOFTMAX, LAYERNORM, ATTENTION, COPY}
	opType := opTypes[r.Intn(len(opTypes))]

	switch opType {
	case UNIFIED_GEMM:
		// Generate realistic GEMM sizes (powers of 2, common transformer dimensions)
		sizes := []int{256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192}
		m := sizes[r.Intn(len(sizes))]
		n := sizes[r.Intn(len(sizes))]
		k := sizes[r.Intn(len(sizes))]

		dtypes := []string{"fp16", "fp32", "bf16", "int8"}
		dtype := dtypes[r.Intn(len(dtypes))]

		return Operation{
			Type:  UNIFIED_GEMM,
			M:     m,
			N:     n,
			K:     k,
			Dtype: dtype,
		}

	case ATTENTION:
		// Sequence lengths common in transformers
		seqLens := []int{128, 256, 512, 1024, 2048, 4096, 8192}
		seqLen := seqLens[r.Intn(len(seqLens))]

		return Operation{
			Type: ATTENTION,
			Size: seqLen,
		}

	default:
		// For other operations, generate reasonable sizes
		size := (r.Intn(8) + 1) * 1024 // 1K to 8K
		return Operation{
			Type: opType,
			Size: size,
		}
	}
}

// OptionsGenerator creates realistic API options
type OptionsGenerator struct{}

func NewOptionsGenerator() *OptionsGenerator {
	return &OptionsGenerator{}
}

func (g *OptionsGenerator) Generate(r *rand.Rand) api.Options {
	// Common context lengths
	contextLengths := []int{512, 1024, 2048, 4096, 8192, 16384, 32768}

	opts := api.Options{
		Runner: api.Runner{
			NumCtx: contextLengths[r.Intn(len(contextLengths))],
		},
	}

	// Sometimes set NumGPU
	if r.Float32() < 0.3 {
		opts.Runner.NumGPU = r.Intn(4) + 1 // 1-4 GPUs
	}

	return opts
}

// Property test runner with shrinking
type PropertyTestRunner struct {
	MaxTests     int
	MaxShrinks   int
	RandomSource *rand.Rand
}

func NewPropertyTestRunner() *PropertyTestRunner {
	return &PropertyTestRunner{
		MaxTests:     1000,
		MaxShrinks:   100,
		RandomSource: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// Property function types for different types
type IntProperty func(int) error
type UintProperty func(uint64) error
type GPUSpecProperty func(*GPUDeviceSpec) error
type ModelConfigProperty func(ModelConfig) error
type WorkloadProperty func([]Operation) error

// RunIntProperty executes a property test with shrinking for int values
func (ptr *PropertyTestRunner) RunIntProperty(
	gen *LayerCountGenerator,
	prop IntProperty,
) error {
	for i := 0; i < ptr.MaxTests; i++ {
		value := gen.Generate(ptr.RandomSource)

		if err := prop(value); err != nil {
			// Property failed, try to shrink
			return ptr.shrinkAndTestInt(gen, prop, value, err)
		}
	}

	return nil // All tests passed
}

func (ptr *PropertyTestRunner) shrinkAndTestInt(
	gen *LayerCountGenerator,
	prop IntProperty,
	failingValue int,
	originalError error,
) error {
	smallestFailing := failingValue
	smallestError := originalError

	candidates := []int{failingValue}

	for shrinkRound := 0; shrinkRound < ptr.MaxShrinks && len(candidates) > 0; shrinkRound++ {
		var nextCandidates []int

		for _, candidate := range candidates {
			if err := prop(candidate); err != nil {
				// Still failing, this is our new smallest
				smallestFailing = candidate
				smallestError = err

				// Try to shrink further
				shrunk := gen.Shrink(candidate)
				nextCandidates = append(nextCandidates, shrunk...)
			}
		}

		candidates = nextCandidates
	}

	return fmt.Errorf("property failed with smallest example %+v: %w", smallestFailing, smallestError)
}

// Add similar methods for other types
func (ptr *PropertyTestRunner) RunUintProperty(
	gen *KVCacheSizeGenerator,
	prop UintProperty,
) error {
	for i := 0; i < ptr.MaxTests; i++ {
		value := gen.Generate(ptr.RandomSource)

		if err := prop(value); err != nil {
			return ptr.shrinkAndTestUint(gen, prop, value, err)
		}
	}
	return nil
}

func (ptr *PropertyTestRunner) shrinkAndTestUint(
	gen *KVCacheSizeGenerator,
	prop UintProperty,
	failingValue uint64,
	originalError error,
) error {
	smallestFailing := failingValue
	smallestError := originalError

	candidates := []uint64{failingValue}

	for shrinkRound := 0; shrinkRound < ptr.MaxShrinks && len(candidates) > 0; shrinkRound++ {
		var nextCandidates []uint64

		for _, candidate := range candidates {
			if err := prop(candidate); err != nil {
				smallestFailing = candidate
				smallestError = err

				shrunk := gen.Shrink(candidate)
				nextCandidates = append(nextCandidates, shrunk...)
			}
		}

		candidates = nextCandidates
	}

	return fmt.Errorf("property failed with smallest example %+v: %w", smallestFailing, smallestError)
}

func (ptr *PropertyTestRunner) RunGPUSpecProperty(
	gen *GPUSpecGenerator,
	prop GPUSpecProperty,
) error {
	for i := 0; i < ptr.MaxTests; i++ {
		value := gen.Generate(ptr.RandomSource)

		if err := prop(value); err != nil {
			return fmt.Errorf("property failed with example %+v: %w", value, err)
		}
	}
	return nil
}

func (ptr *PropertyTestRunner) RunModelConfigProperty(
	gen *ModelConfigGenerator,
	prop ModelConfigProperty,
) error {
	for i := 0; i < ptr.MaxTests; i++ {
		value := gen.Generate(ptr.RandomSource)

		if err := prop(value); err != nil {
			return fmt.Errorf("property failed with example %+v: %w", value, err)
		}
	}
	return nil
}

func (ptr *PropertyTestRunner) RunWorkloadProperty(
	gen *WorkloadGenerator,
	prop WorkloadProperty,
) error {
	for i := 0; i < ptr.MaxTests; i++ {
		value := gen.Generate(ptr.RandomSource)

		if err := prop(value); err != nil {
			return fmt.Errorf("property failed with example %+v: %w", value, err)
		}
	}
	return nil
}

// Composite generators for complex scenarios
type ModelConfigGenerator struct {
	layerGen *LayerCountGenerator
	kvGen    *KVCacheSizeGenerator
	optsGen  *OptionsGenerator
}

func NewModelConfigGenerator() *ModelConfigGenerator {
	return &ModelConfigGenerator{
		layerGen: NewLayerCountGenerator(),
		kvGen:    NewKVCacheSizeGenerator(),
		optsGen:  NewOptionsGenerator(),
	}
}

type ModelConfig struct {
	Layers int
	KVSize uint64
	Options api.Options
}

func (g *ModelConfigGenerator) Generate(r *rand.Rand) ModelConfig {
	return ModelConfig{
		Layers:  g.layerGen.Generate(r),
		KVSize:  g.kvGen.Generate(r),
		Options: g.optsGen.Generate(r),
	}
}

func (g *ModelConfigGenerator) Shrink(value ModelConfig) []ModelConfig {
	shrunk := make([]ModelConfig, 0, 10)

	// Shrink layers
	for _, layers := range g.layerGen.Shrink(value.Layers) {
		smaller := value
		smaller.Layers = layers
		shrunk = append(shrunk, smaller)
	}

	// Shrink KV size
	for _, kvSize := range g.kvGen.Shrink(value.KVSize) {
		smaller := value
		smaller.KVSize = kvSize
		shrunk = append(shrunk, smaller)
	}

	// Shrink context length
	if value.Options.Runner.NumCtx > 512 {
		smaller := value
		smaller.Options.Runner.NumCtx = value.Options.Runner.NumCtx / 2
		shrunk = append(shrunk, smaller)
	}

	return shrunk
}