package llm

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
)

// ExpertState represents the current state of an expert
type ExpertState int

const (
	ExpertStateUnloaded ExpertState = iota // Expert not loaded in memory
	ExpertStateLoading                     // Expert currently being loaded
	ExpertStateActive                      // Expert loaded and active
	ExpertStateInactive                    // Expert loaded but not actively used
	ExpertStateOffloading                  // Expert being moved to CPU/disk
)

func (e ExpertState) String() string {
	switch e {
	case ExpertStateUnloaded:
		return "unloaded"
	case ExpertStateLoading:
		return "loading"
	case ExpertStateActive:
		return "active"
	case ExpertStateInactive:
		return "inactive"
	case ExpertStateOffloading:
		return "offloading"
	default:
		return "unknown"
	}
}

// ExpertInfo tracks information about individual experts
type ExpertInfo struct {
	ID          int                    // Expert ID
	LayerIndex  int                    // Layer this expert belongs to
	State       ExpertState            // Current state
	LastUsed    time.Time              // Last access time
	AccessCount uint64                 // Number of times accessed
	VRAMSize    uint64                 // Size when loaded in VRAM
	CPUSize     uint64                 // Size when stored in CPU memory
	Weights     map[string]interface{} // Expert weight tensors
	Location    ExpertLocation          // Current storage location
	mu          sync.RWMutex           // Protects concurrent access
}

// ExpertLocation indicates where expert weights are currently stored
type ExpertLocation int

const (
	LocationVRAM ExpertLocation = iota // Expert weights in GPU VRAM
	LocationCPU                        // Expert weights in CPU memory
	LocationDisk                       // Expert weights on disk/storage
)

func (l ExpertLocation) String() string {
	switch l {
	case LocationVRAM:
		return "vram"
	case LocationCPU:
		return "cpu"
	case LocationDisk:
		return "disk"
	default:
		return "unknown"
	}
}

// MoEOptimizer manages memory-efficient loading of MoE experts
type MoEOptimizer struct {
	experts           map[int]*ExpertInfo // Expert ID -> Expert info
	activeExperts     map[int]bool        // Currently active experts
	expertsByLayer    map[int][]int       // Layer -> Expert IDs
	maxActiveExperts  int                 // Maximum experts to keep active
	maxVRAMExperts    int                 // Maximum experts in VRAM
	totalExperts      int                 // Total number of experts
	totalLayers       int                 // Total number of layers
	expertUsageWindow time.Duration       // Time window for expert usage tracking

	// Memory management
	vramBudget        uint64              // Available VRAM for experts
	cpuBudget         uint64              // Available CPU memory for experts
	currentVRAMUsage  uint64              // Current VRAM usage by experts
	currentCPUUsage   uint64              // Current CPU usage by experts

	// Configuration
	enableDynamicLoading bool            // Enable dynamic expert loading
	enableOffloading     bool            // Enable expert offloading
	enableSparseCompute  bool            // Enable sparse computation optimization

	// Concurrency control
	mu                sync.RWMutex       // Protects optimizer state
	loadingExperts    map[int]*sync.Mutex // Prevents concurrent loading of same expert

	// Performance metrics
	loadTime          time.Duration       // Total time spent loading experts
	offloadTime       time.Duration       // Total time spent offloading experts
	cacheHits         uint64              // Number of cache hits
	cacheMisses       uint64              // Number of cache misses

}

// MoEOptimizerConfig contains configuration for the MoE optimizer
type MoEOptimizerConfig struct {
	EnableDynamicLoading bool          // Enable dynamic expert loading
	EnableOffloading     bool          // Enable expert offloading to CPU/disk
	EnableSparseCompute  bool          // Enable sparse computation optimizations
	MaxActiveExperts     int           // Maximum experts to keep active (0 = auto)
	MaxVRAMExperts       int           // Maximum experts in VRAM (0 = auto)
	VRAMBudget          uint64         // VRAM budget for experts (0 = auto)
	CPUBudget           uint64         // CPU memory budget for experts (0 = auto)
	ExpertUsageWindow   time.Duration  // Time window for tracking expert usage
}

// DefaultMoEOptimizerConfig returns default configuration for MoE optimization
func DefaultMoEOptimizerConfig() MoEOptimizerConfig {
	return MoEOptimizerConfig{
		EnableDynamicLoading: true,
		EnableOffloading:     true,
		EnableSparseCompute:  true,
		MaxActiveExperts:     0, // Auto-determined
		MaxVRAMExperts:       0, // Auto-determined
		VRAMBudget:          0, // Auto-determined
		CPUBudget:           0, // Auto-determined
		ExpertUsageWindow:   5 * time.Minute,
	}
}

// NewMoEOptimizer creates a new MoE optimizer for the given model
func NewMoEOptimizer(ggml *ggml.GGML, config MoEOptimizerConfig) *MoEOptimizer {
	if ggml == nil {
		return nil
	}

	kv := ggml.KV()

	// Try different MoE architectures' key naming conventions
	// Note: GGML keyValue automatically adds architecture prefix, so we only need the key suffix
	totalExperts := int(kv.Uint("expert_count", 0))
	totalLayers := int(kv.BlockCount())
	expertsPerToken := int(kv.Uint("expert_used_count", 2))

	if totalExperts == 0 {
		return nil
	}

	optimizer := &MoEOptimizer{
		experts:              make(map[int]*ExpertInfo),
		activeExperts:        make(map[int]bool),
		expertsByLayer:       make(map[int][]int),
		totalExperts:         totalExperts,
		totalLayers:          totalLayers,
		expertUsageWindow:    config.ExpertUsageWindow,
		enableDynamicLoading: config.EnableDynamicLoading,
		enableOffloading:     config.EnableOffloading,
		enableSparseCompute:  config.EnableSparseCompute,
		loadingExperts:       make(map[int]*sync.Mutex),
	}

	// Auto-configure limits if not specified
	if config.MaxActiveExperts <= 0 {
		// Keep 2-4 experts per token active, depending on model size
		optimizer.maxActiveExperts = max(expertsPerToken*2, min(totalExperts/4, expertsPerToken*4))
	} else {
		optimizer.maxActiveExperts = config.MaxActiveExperts
	}

	if config.MaxVRAMExperts <= 0 {
		// Keep enough experts in VRAM for smooth operation
		optimizer.maxVRAMExperts = max(expertsPerToken*3, min(totalExperts/2, expertsPerToken*6))
	} else {
		optimizer.maxVRAMExperts = config.MaxVRAMExperts
	}

	// Initialize expert tracking
	optimizer.initializeExperts(ggml)


	return optimizer
}

// initializeExperts scans the model and initializes expert tracking
func (m *MoEOptimizer) initializeExperts(ggml *ggml.GGML) {
	// For models where we already know the total expert count, 
	// initialize all experts directly
	if m.totalExperts > 0 {
		
		for expertID := 0; expertID < m.totalExperts; expertID++ {
			expert := &ExpertInfo{
				ID:         expertID,
				LayerIndex: 0, // For Granite MoE, experts are global across layers
				State:      ExpertStateUnloaded,
				LastUsed:   time.Now(),
				Weights:    make(map[string]interface{}),
				Location:   LocationDisk,
			}

			// Estimate expert size (in production, this would be calculated from actual tensors)
			expert.VRAMSize = 100 * 1024 * 1024  // 100MB per expert
			expert.CPUSize = expert.VRAMSize

			m.experts[expertID] = expert
			m.expertsByLayer[0] = append(m.expertsByLayer[0], expertID)
		}
		return
	}

	// Fallback: scan tensors to find experts
	tensors := ggml.Tensors()
	layers := tensors.GroupLayers()

	expertCount := 0
	for layerName, layer := range layers {
		layerIndex := -1
		if _, err := fmt.Sscanf(layerName, "blk.%d", &layerIndex); err != nil {
			continue
		}

		// Look for expert tensors in this layer
		var expertTensors []string
		for tensorName := range layer {
			if containsExpertPattern(tensorName) {
				expertTensors = append(expertTensors, tensorName)
			}
		}

		if len(expertTensors) == 0 {
			continue
		}

		// Determine number of experts in this layer
		expertsInLayer := m.countExpertsInLayer(expertTensors)

		for expertID := 0; expertID < expertsInLayer; expertID++ {
			globalExpertID := expertCount + expertID

			expert := &ExpertInfo{
				ID:         globalExpertID,
				LayerIndex: layerIndex,
				State:      ExpertStateUnloaded,
				LastUsed:   time.Now(),
				Weights:    make(map[string]interface{}),
				Location:   LocationDisk,
			}

			// Collect expert-specific tensors
			for tensorName, tensor := range layer {
				if m.isExpertTensor(tensorName, expertID) {
					expert.Weights[tensorName] = tensor
					expert.VRAMSize += tensor.Size()
					expert.CPUSize += tensor.Size()
				}
			}

			m.experts[globalExpertID] = expert
			m.expertsByLayer[layerIndex] = append(m.expertsByLayer[layerIndex], globalExpertID)
		}

		expertCount += expertsInLayer
	}

}

// containsExpertPattern checks if a tensor name contains expert-related patterns
func containsExpertPattern(name string) bool {
	patterns := []string{
		"ffn_gate_exps",
		"ffn_up_exps",
		"ffn_down_exps",
		"ffn_norm_exps",
		"experts.",
		".ffn_gate_exps.",
		".ffn_up_exps.",
		".ffn_down_exps.",
	}

	for _, pattern := range patterns {
		if strings.Contains(name, pattern) {
			return true
		}
	}
	return false
}

// countExpertsInLayer determines the number of experts in a layer based on tensor patterns
func (m *MoEOptimizer) countExpertsInLayer(tensorNames []string) int {
	// For Granite MoE, expert tensors have shape like [512 1536 40] where 40 is the expert count
	// For traditional MoE, look for expert indices in tensor names

	// First try to find expert count from tensor shapes (Granite MoE style)
	for _, name := range tensorNames {
		if strings.Contains(name, "ffn_gate_exps") || strings.Contains(name, "ffn_up_exps") || strings.Contains(name, "ffn_down_exps") {
			// For Granite MoE, the expert count is specified in the total experts
			return m.totalExperts
		}
	}

	// Fallback to traditional expert indexing
	maxExpertIndex := -1
	for _, name := range tensorNames {
		// Extract expert index from tensor name
		// Pattern: "ffn_gate_exps.weight" or "experts.0.w1.weight"
		for i := 0; i < m.totalExperts; i++ {
			if m.isExpertTensor(name, i) {
				maxExpertIndex = max(maxExpertIndex, i)
			}
		}
	}

	if maxExpertIndex >= 0 {
		return maxExpertIndex + 1
	}

	return 0
}

// isExpertTensor checks if a tensor belongs to a specific expert
func (m *MoEOptimizer) isExpertTensor(tensorName string, expertID int) bool {
	// For Granite MoE, all experts are in the same tensor
	// The expert dimension is the last dimension of the tensor
	if strings.Contains(tensorName, "ffn_gate_exps") ||
	   strings.Contains(tensorName, "ffn_up_exps") ||
	   strings.Contains(tensorName, "ffn_down_exps") {
		// For Granite MoE, consider all expert IDs valid for expert tensors
		return expertID >= 0 && expertID < m.totalExperts
	}

	// Handle traditional MoE expert tensor naming patterns
	patterns := []string{
		fmt.Sprintf("experts.%d.", expertID),
		fmt.Sprintf("ffn_gate_exps.%d", expertID),
		fmt.Sprintf("ffn_up_exps.%d", expertID),
		fmt.Sprintf("ffn_down_exps.%d", expertID),
	}

	for _, pattern := range patterns {
		if strings.Contains(tensorName, pattern) {
			return true
		}
	}

	return false
}

// RequestExperts is called when the model needs specific experts for computation
func (m *MoEOptimizer) RequestExperts(ctx context.Context, expertIDs []int, priority int) error {
	if !m.enableDynamicLoading {
		return nil
	}


	// Reject empty requests
	if len(expertIDs) == 0 {
		return nil
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	start := time.Now()
	defer func() {
		m.loadTime += time.Since(start)
	}()

	var toLoad []int
	var hits, misses int

	// Check which experts need to be loaded
	for _, expertID := range expertIDs {
		expert, exists := m.experts[expertID]
		if !exists {
			continue
		}

		expert.mu.Lock()
		expert.LastUsed = time.Now()
		expert.AccessCount++

		if expert.State == ExpertStateActive {
			hits++
			m.activeExperts[expertID] = true
		} else {
			misses++
			toLoad = append(toLoad, expertID)
		}
		expert.mu.Unlock()
	}

	m.cacheHits += uint64(hits)
	m.cacheMisses += uint64(misses)


	if len(toLoad) == 0 {
		return nil
	}

	// Ensure we have space for new experts
	if err := m.makeSpaceForExperts(len(toLoad)); err != nil {
		return fmt.Errorf("failed to make space for experts: %w", err)
	}

	// Load required experts
	var loadedCount int
	for _, expertID := range toLoad {
		if err := m.loadExpert(ctx, expertID); err != nil {
			continue
		}
		m.activeExperts[expertID] = true
		loadedCount++
	}

	return nil
}

// makeSpaceForExperts ensures there's space for the requested number of experts
func (m *MoEOptimizer) makeSpaceForExperts(count int) error {
	currentActive := len(m.activeExperts)

	// Check if we need to evict experts
	toEvict := (currentActive + count) - m.maxActiveExperts
	if toEvict <= 0 {
		return nil
	}

	// Find least recently used experts to evict
	candidates := m.getLRUExperts(toEvict)

	for _, expertID := range candidates {
		if err := m.evictExpert(expertID); err != nil {
			continue
		}
		delete(m.activeExperts, expertID)
	}

	return nil
}

// getLRUExperts returns the least recently used experts for eviction
func (m *MoEOptimizer) getLRUExperts(count int) []int {
	type expertUsage struct {
		id       int
		lastUsed time.Time
	}

	var candidates []expertUsage
	for expertID := range m.activeExperts {
		expert := m.experts[expertID]
		expert.mu.RLock()
		candidates = append(candidates, expertUsage{
			id:       expertID,
			lastUsed: expert.LastUsed,
		})
		expert.mu.RUnlock()
	}

	// Sort by last used time (oldest first)
	for i := 0; i < len(candidates)-1; i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[i].lastUsed.After(candidates[j].lastUsed) {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	result := make([]int, min(count, len(candidates)))
	for i := 0; i < len(result); i++ {
		result[i] = candidates[i].id
	}

	return result
}

// loadExpert loads an expert into active memory
func (m *MoEOptimizer) loadExpert(ctx context.Context, expertID int) error {
	expert, exists := m.experts[expertID]
	if !exists {
		return fmt.Errorf("expert %d not found", expertID)
	}

	expert.mu.Lock()
	defer expert.mu.Unlock()

	if expert.State == ExpertStateActive {
		return nil // Already loaded
	}

	// Prevent concurrent loading of the same expert
	if _, exists := m.loadingExperts[expertID]; !exists {
		m.loadingExperts[expertID] = &sync.Mutex{}
	}

	expertMutex := m.loadingExperts[expertID]
	expertMutex.Lock()
	defer expertMutex.Unlock()

	expert.State = ExpertStateLoading


	// Simulate expert loading based on current location
	switch expert.Location {
	case LocationVRAM:
		// Already in VRAM, just activate
		expert.State = ExpertStateActive
	case LocationCPU:
		// Move from CPU to VRAM
		if err := m.moveExpertToVRAM(expert); err != nil {
			expert.State = ExpertStateInactive
			return err
		}
		expert.State = ExpertStateActive
	case LocationDisk:
		// Load from disk to VRAM
		if err := m.loadExpertFromDisk(expert); err != nil {
			expert.State = ExpertStateUnloaded
			return err
		}
		expert.State = ExpertStateActive
	}

	m.currentVRAMUsage += expert.VRAMSize
	expert.Location = LocationVRAM
	expert.LastUsed = time.Now()

	return nil
}

// evictExpert removes an expert from active memory
func (m *MoEOptimizer) evictExpert(expertID int) error {
	expert, exists := m.experts[expertID]
	if !exists {
		return fmt.Errorf("expert %d not found", expertID)
	}

	expert.mu.Lock()
	defer expert.mu.Unlock()

	if expert.State != ExpertStateActive {
		return nil // Not active
	}

	expert.State = ExpertStateOffloading


	// Choose eviction strategy based on configuration
	if m.enableOffloading {
		// Move to CPU memory if possible
		if m.currentCPUUsage+expert.CPUSize <= m.cpuBudget {
			if err := m.moveExpertToCPU(expert); err != nil {
				m.moveExpertToDisk(expert)
			}
		} else {
			m.moveExpertToDisk(expert)
		}
	} else {
		// Simply mark as inactive but keep in VRAM
		expert.State = ExpertStateInactive
		return nil
	}

	m.currentVRAMUsage -= expert.VRAMSize
	expert.State = ExpertStateUnloaded

	return nil
}

// moveExpertToVRAM moves an expert from CPU to VRAM
func (m *MoEOptimizer) moveExpertToVRAM(expert *ExpertInfo) error {
	// Simulate memory transfer
	time.Sleep(time.Microsecond * 100) // Simulate transfer latency

	m.currentCPUUsage -= expert.CPUSize
	expert.Location = LocationVRAM

	return nil
}

// moveExpertToCPU moves an expert from VRAM to CPU memory
func (m *MoEOptimizer) moveExpertToCPU(expert *ExpertInfo) error {
	// Simulate memory transfer
	time.Sleep(time.Microsecond * 200) // Simulate transfer latency

	m.currentCPUUsage += expert.CPUSize
	expert.Location = LocationCPU

	return nil
}

// moveExpertToDisk moves an expert to disk storage
func (m *MoEOptimizer) moveExpertToDisk(expert *ExpertInfo) error {
	// Simulate disk write
	time.Sleep(time.Millisecond * 5) // Simulate disk latency

	if expert.Location == LocationCPU {
		m.currentCPUUsage -= expert.CPUSize
	}
	expert.Location = LocationDisk

	return nil
}

// loadExpertFromDisk loads an expert from disk storage
func (m *MoEOptimizer) loadExpertFromDisk(expert *ExpertInfo) error {
	// Simulate disk read
	time.Sleep(time.Millisecond * 10) // Simulate disk latency

	expert.Location = LocationVRAM

	return nil
}

// GetOptimizationStats returns current optimization statistics
func (m *MoEOptimizer) GetOptimizationStats() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()




	// Handle division by zero for hit rate
	var hitRate float64
	if m.cacheHits+m.cacheMisses > 0 {
		hitRate = float64(m.cacheHits) / float64(m.cacheHits+m.cacheMisses)
	} else {
		hitRate = 0.0
	}

	stats := map[string]interface{}{
		"moe_optimization":   true,
		"total_experts":      m.totalExperts,
		"active_experts":     len(m.activeExperts),
		"max_active":         m.maxActiveExperts,
		"max_vram":          m.maxVRAMExperts,
		"vram_usage":        format.HumanBytes2(m.currentVRAMUsage),
		"cpu_usage":         format.HumanBytes2(m.currentCPUUsage),
		"vram_budget":       format.HumanBytes2(m.vramBudget),
		"cpu_budget":        format.HumanBytes2(m.cpuBudget),
		"cache_hits":        m.cacheHits,
		"cache_misses":      m.cacheMisses,
		"hit_rate":          hitRate,
		"total_load_time":   m.loadTime.String(),
		"total_offload_time": m.offloadTime.String(),
		"dynamic_loading":   m.enableDynamicLoading,
		"offloading":        m.enableOffloading,
		"sparse_compute":    m.enableSparseCompute,
	}


	// Expert location distribution
	locationCounts := map[ExpertLocation]int{}
	for _, expert := range m.experts {
		expert.mu.RLock()
		locationCounts[expert.Location]++
		expert.mu.RUnlock()
	}

	stats["experts_in_vram"] = locationCounts[LocationVRAM]
	stats["experts_in_cpu"] = locationCounts[LocationCPU]
	stats["experts_on_disk"] = locationCounts[LocationDisk]

	return stats
}

// UpdateMemoryBudgets updates the memory budgets for expert management
func (m *MoEOptimizer) UpdateMemoryBudgets(vramBudget, cpuBudget uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.vramBudget = vramBudget
	m.cpuBudget = cpuBudget
}

// CleanupInactiveExperts removes experts that haven't been used recently
func (m *MoEOptimizer) CleanupInactiveExperts() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	cutoff := time.Now().Add(-m.expertUsageWindow)
	var toCleanup []int

	for expertID, expert := range m.experts {
		expert.mu.RLock()
		if expert.LastUsed.Before(cutoff) && expert.State != ExpertStateActive {
			toCleanup = append(toCleanup, expertID)
		}
		expert.mu.RUnlock()
	}

	for _, expertID := range toCleanup {
		m.evictExpert(expertID)
		delete(m.activeExperts, expertID)
	}

	if len(toCleanup) > 0 {
	}
	
	return nil
}

// UpdatePerformanceMetrics updates real-time performance metrics
func (m *MoEOptimizer) UpdatePerformanceMetrics(tokensPerSecond float64, gpuUtilization map[string]float64) {
	if !m.IsEnabled() {
		return
	}

	// This would update internal performance tracking
}

// IsEnabled returns whether the MoE optimizer is enabled and functional
func (m *MoEOptimizer) IsEnabled() bool {
	if m == nil {
		return false
	}
	return m.enableDynamicLoading || m.enableOffloading || m.enableSparseCompute
}

// Close cleans up the MoE optimizer and releases resources
func (m *MoEOptimizer) Close() {
	if m == nil {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Clean up all experts
	for expertID := range m.activeExperts {
		m.evictExpert(expertID)
	}

	m.experts = make(map[int]*ExpertInfo)
	m.activeExperts = make(map[int]bool)

}

// Global MoE optimizer management
var globalMoEOptimizer *MoEOptimizer
var globalMoEMutex sync.RWMutex

// SetGlobalMoEOptimizer sets the global MoE optimizer instance
func SetGlobalMoEOptimizer(optimizer *MoEOptimizer) {
	globalMoEMutex.Lock()
	defer globalMoEMutex.Unlock()

	if optimizer != nil {
	}

	if globalMoEOptimizer != nil {
		globalMoEOptimizer.Close()
	}
	globalMoEOptimizer = optimizer
}

// GetGlobalMoEOptimizer returns the current global MoE optimizer
func GetGlobalMoEOptimizer() *MoEOptimizer {
	globalMoEMutex.RLock()
	defer globalMoEMutex.RUnlock()

	return globalMoEOptimizer
}


// CleanupGlobalMoEOptimizer cleans up the global MoE optimizer
func CleanupGlobalMoEOptimizer() {
	globalMoEMutex.Lock()
	defer globalMoEMutex.Unlock()

	if globalMoEOptimizer != nil {
		globalMoEOptimizer.Close()
		globalMoEOptimizer = nil
	}
}