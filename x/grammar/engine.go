//go:build mlx

package grammar

import (
	"container/list"
	"fmt"
	"math"
	"sync"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// maskCache provides LRU caching for computed masks.
type maskCache struct {
	cache   map[uint64]*list.Element
	order   *list.List
	maxSize int
	mu      sync.Mutex
}

type maskEntry struct {
	sig  uint64
	mask *mlx.Array
}

// newMaskCache creates a new mask cache with the given max size
// If maxSize <= 0, the cache is disabled (Get/Put are no-ops)
func newMaskCache(maxSize int) *maskCache {
	if maxSize <= 0 {
		return &maskCache{
			cache:   make(map[uint64]*list.Element),
			order:   list.New(),
			maxSize: 0, // Signals disabled
		}
	}
	return &maskCache{
		cache:   make(map[uint64]*list.Element),
		order:   list.New(),
		maxSize: maxSize,
	}
}

// get retrieves a cached mask, returning nil if not found.
// Updates LRU order on cache hit.
func (c *maskCache) get(sig uint64) *mlx.Array {
	if c.maxSize <= 0 {
		return nil // Cache disabled
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if elem, ok := c.cache[sig]; ok {
		c.order.MoveToFront(elem)
		return elem.Value.(*maskEntry).mask
	}
	return nil
}

// put stores a mask in the cache with LRU eviction.
func (c *maskCache) put(sig uint64, mask *mlx.Array) {
	if c.maxSize <= 0 {
		return // Cache disabled
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, exists := c.cache[sig]; exists {
		c.order.MoveToFront(elem)
		return
	}

	// Evict oldest if at capacity (safe since maxSize > 0)
	if c.order.Len() >= c.maxSize {
		oldest := c.order.Back()
		if oldest != nil {
			entry := oldest.Value.(*maskEntry)
			entry.mask.Free()
			delete(c.cache, entry.sig)
			c.order.Remove(oldest)
		}
	}

	elem := c.order.PushFront(&maskEntry{sig: sig, mask: mask})
	c.cache[sig] = elem
}

// clear frees all cached masks.
func (c *maskCache) clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for elem := c.order.Front(); elem != nil; elem = elem.Next() {
		elem.Value.(*maskEntry).mask.Free()
	}
	c.cache = make(map[uint64]*list.Element)
	c.order.Init()
}

// size returns the number of cached masks.
func (c *maskCache) size() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.cache)
}

// Engine applies grammar constraints to model outputs using MLX.
// It uses a token→pda bridge for strict correctness with arbitrary BPE tokens.
type Engine struct {
	// The compiled grammar
	grammar *Grammar

	// bridge for token validation
	bridge   *bridge
	analyzer *analyzer

	// Current parser state (configSet for nondeterminism)
	configSet *configSet

	// Token vocabulary from the model
	vocab     []string
	tokenToID map[string]int // O(1) lookup for AcceptString

	// Mask cache: configSig → valid token mask (LRU)
	maskCache *maskCache

	// Cached negative infinity mask for invalid tokens
	negInfMask *mlx.Array

	// Threshold for comparison (0.5 since mask values are 0 or 1)
	threshold *mlx.Array

	// Vocabulary size
	vocabSize int32

	// Reusable buffers for candidate filtering (avoid allocations)
	candidateMark []bool // indexed by tokenID, true if in candidate set
	touched       []int  // tokenIDs that were marked (for reset)
	dpCandidates  []int  // candidates requiring DP validation

	// Reusable buffer for valid token indices (for GPU scatter)
	validTokenIDs []int32
}

// EngineOption configures an Engine
type EngineOption func(*Engine)

// WithMaskCacheSize sets the mask cache size (default 1024)
func WithMaskCacheSize(size int) EngineOption {
	return func(e *Engine) {
		e.maskCache = newMaskCache(size)
	}
}

// NewEngine creates a new constrained decoding engine.
// grammar is the compiled grammar (use JSONGrammar() or ParseEBNF()).
// vocab is the list of token strings from the model's tokenizer.
func NewEngine(grammar *Grammar, vocab []string, opts ...EngineOption) (*Engine, error) {
	if grammar == nil {
		return nil, fmt.Errorf("grammar cannot be nil")
	}

	// Build analyzer and bridge
	analyzer := newAnalyzer(vocab, grammar.matcher)
	bridge := newBridge(grammar.pda, analyzer)

	// Initialize config set from pda initial state
	initialConfig := newConfigSet(grammar.pda.StartState, nil)

	// Build token lookup map for O(1) AcceptString
	tokenToID := make(map[string]int, len(vocab))
	for i, tok := range vocab {
		tokenToID[tok] = i
	}

	e := &Engine{
		grammar:       grammar,
		bridge:        bridge,
		analyzer:      analyzer,
		configSet:     initialConfig,
		vocab:         vocab,
		tokenToID:     tokenToID,
		maskCache:     newMaskCache(1024),
		vocabSize:     int32(len(vocab)),
		candidateMark: make([]bool, len(vocab)),
		touched:       make([]int, 0, 10000),
		validTokenIDs: make([]int32, 0, 10000),
	}

	// Apply options
	for _, opt := range opts {
		opt(e)
	}

	// Create the negative infinity mask and threshold
	if e.vocabSize > 0 {
		e.negInfMask = mlx.FullDtype(float32(math.Inf(-1)), mlx.DtypeFloat32, e.vocabSize)
		mlx.Keep(e.negInfMask)

		e.threshold = mlx.NewScalarArray(0.5)
		mlx.Keep(e.threshold)
	}

	return e, nil
}

// ApplyMask applies grammar constraints to logits.
// Returns logits with invalid tokens set to -inf.
func (e *Engine) ApplyMask(logits *mlx.Array) *mlx.Array {
	sig := e.configSet.signature()

	// Check state cache first (exact state match)
	if cached := e.maskCache.get(sig); cached != nil {
		condition := mlx.GreaterEqual(cached, e.threshold)
		return mlx.Where(condition, logits, e.negInfMask)
	}

	// Compute valid tokens using candidate filtering:
	// 1. Get valid terminal IDs from current grammar state
	// 2. Get candidate tokens (those that START with valid terminals)
	// 3. Run DP validation only on candidates
	// This is O(candidates) instead of O(vocab_size)

	validTerminalIDs := e.bridge.validTerminalIDs(e.configSet)

	// Use pre-partitioned token groups for fast candidate building
	// This eliminates per-token branching - just direct slice appends
	e.validTokenIDs = e.validTokenIDs[:0]
	e.dpCandidates = e.dpCandidates[:0]
	e.touched = e.touched[:0]

	for _, tid := range validTerminalIDs {
		groups := e.analyzer.terminalGroups(tid)

		// Direct append of exact matches (no per-token check needed)
		e.validTokenIDs = append(e.validTokenIDs, groups.ExactMatches...)

		// Collect DP candidates (may have duplicates across terminals)
		for _, tokenID := range groups.DPCandidates {
			if !e.candidateMark[tokenID] {
				e.candidateMark[tokenID] = true
				e.dpCandidates = append(e.dpCandidates, tokenID)
				e.touched = append(e.touched, tokenID)
			}
		}
	}

	// Reset marks for next call
	for _, id := range e.touched {
		e.candidateMark[id] = false
	}

	for _, tokenID := range e.dpCandidates {
		if e.bridge.IsTokenValid(tokenID, e.configSet) {
			e.validTokenIDs = append(e.validTokenIDs, int32(tokenID))
		}
	}

	// Create and cache the mask on GPU using index updates
	mask := mlx.Zeros([]int32{e.vocabSize})
	if len(e.validTokenIDs) > 0 {
		indices := mlx.NewArrayInt32(e.validTokenIDs, []int32{int32(len(e.validTokenIDs))})
		values := mlx.Ones(int32(len(e.validTokenIDs)))
		mask = mlx.PutAlongAxis(mask, indices, values, 0)
	}
	mlx.Keep(mask)

	// Cache by state signature
	e.maskCache.put(sig, mask)

	// Apply mask
	condition := mlx.GreaterEqual(mask, e.threshold)
	return mlx.Where(condition, logits, e.negInfMask)
}

// Accept processes a token and updates the parser state.
// Returns true if the token was valid and accepted.
func (e *Engine) Accept(tokenID int) bool {
	if tokenID < 0 || tokenID >= len(e.vocab) {
		return false
	}

	newConfig := e.bridge.acceptToken(tokenID, e.configSet)
	if newConfig == nil {
		return false
	}
	e.configSet = newConfig
	return true
}

// AcceptString processes a token string directly.
// Returns true if the token was valid and accepted.
func (e *Engine) AcceptString(token string) bool {
	if id, ok := e.tokenToID[token]; ok {
		return e.Accept(id)
	}
	return false
}

// IsComplete returns true if the current state is accepting.
func (e *Engine) IsComplete() bool {
	return e.bridge.isAccepting(e.configSet)
}

// Reset resets the engine to initial state.
func (e *Engine) Reset() {
	e.configSet = newConfigSet(e.grammar.pda.StartState, nil)
}

// validTokens returns the indices of tokens that are currently valid.
func (e *Engine) validTokens() []int {
	return e.bridge.validTokens(e.configSet)
}

// validTerminals returns the valid terminal patterns from the current state.
func (e *Engine) validTerminals() []string {
	return e.bridge.validTerminals(e.configSet)
}

// Close releases MLX resources.
func (e *Engine) Close() {
	if e.maskCache != nil {
		e.maskCache.clear()
	}
	if e.negInfMask != nil {
		e.negInfMask.Free()
	}
	if e.threshold != nil {
		e.threshold.Free()
	}
}
