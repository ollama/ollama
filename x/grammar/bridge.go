//go:build mlx

package grammar

import (
	"encoding/binary"
	"hash/fnv"
	"sort"
	"sync"
)

// visitedMapPool reduces allocations for visited maps in bridge operations
var visitedMapPool = sync.Pool{
	New: func() interface{} {
		return make(map[stateStackKey]bool, 16)
	},
}

// getVisitedMap gets a map from the pool
func getVisitedMap() map[stateStackKey]bool {
	return visitedMapPool.Get().(map[stateStackKey]bool)
}

// putVisitedMap returns a map to the pool after clearing it
func putVisitedMap(m map[stateStackKey]bool) {
	for k := range m {
		delete(m, k)
	}
	visitedMapPool.Put(m)
}

// parserConfig represents a pda state+stack combination
type parserConfig struct {
	state state
	Stack []stackSymbol
}

// clone creates a deep copy of the config
func (c *parserConfig) clone() *parserConfig {
	newStack := make([]stackSymbol, len(c.Stack))
	copy(newStack, c.Stack)
	return &parserConfig{
		state: c.state,
		Stack: newStack,
	}
}

// key returns a unique key for this config for deduplication
func (c *parserConfig) key() uint64 {
	h := fnv.New64a()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(c.state))
	h.Write(buf[:])
	for _, sym := range c.Stack {
		binary.LittleEndian.PutUint64(buf[:], uint64(sym))
		h.Write(buf[:])
	}
	return h.Sum64()
}

// configSet represents a set of parser configurations (for nondeterminism)
type configSet struct {
	configs    []*parserConfig
	normalized bool   // true if already deduplicated and sorted
	cachedSig  uint64 // cached signature after normalization
}

// newConfigSet creates a new config set with a single configuration
func newConfigSet(state state, stack []stackSymbol) *configSet {
	return &configSet{
		configs: []*parserConfig{
			{state: state, Stack: stack},
		},
		normalized: true, // single config is already normalized
	}
}

// normalize deduplicates and sorts configs for stable signatures
func (c *configSet) normalize() {
	if c.normalized || len(c.configs) <= 1 {
		c.normalized = true
		return
	}

	// Deduplicate using a map
	seen := make(map[uint64]*parserConfig, len(c.configs))
	for _, cfg := range c.configs {
		key := cfg.key()
		if _, exists := seen[key]; !exists {
			seen[key] = cfg
		}
	}

	// Extract unique configs
	unique := make([]*parserConfig, 0, len(seen))
	for _, cfg := range seen {
		unique = append(unique, cfg)
	}

	// Sort by key for deterministic ordering
	sort.Slice(unique, func(i, j int) bool {
		return unique[i].key() < unique[j].key()
	})

	c.configs = unique
	c.normalized = true
}

// signature returns a hash for cache lookup (normalizes first)
func (c *configSet) signature() uint64 {
	c.normalize()

	// Return cached signature if available
	if c.cachedSig != 0 {
		return c.cachedSig
	}

	h := fnv.New64a()

	// Hash number of configs
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(len(c.configs)))
	h.Write(buf[:])

	// Hash each config (already sorted)
	for _, cfg := range c.configs {
		binary.LittleEndian.PutUint64(buf[:], uint64(cfg.state))
		h.Write(buf[:])

		binary.LittleEndian.PutUint64(buf[:], uint64(len(cfg.Stack)))
		h.Write(buf[:])

		for _, sym := range cfg.Stack {
			binary.LittleEndian.PutUint64(buf[:], uint64(sym))
			h.Write(buf[:])
		}
	}

	c.cachedSig = h.Sum64()
	return c.cachedSig
}

// isEmpty returns true if there are no configurations
func (c *configSet) isEmpty() bool {
	return len(c.configs) == 0
}

// clone creates a deep copy of the config set
func (c *configSet) clone() *configSet {
	newConfigs := make([]*parserConfig, len(c.configs))
	for i, cfg := range c.configs {
		newConfigs[i] = cfg.clone()
	}
	return &configSet{configs: newConfigs}
}

// bridge connects token analysis to pda validation
type bridge struct {
	pda      *pda
	analyzer *analyzer
}

// newBridge creates a new bridge
func newBridge(pda *pda, analyzer *analyzer) *bridge {
	return &bridge{
		pda:      pda,
		analyzer: analyzer,
	}
}

// IsTokenValid checks if token T can be consumed from the current config
// This is the main entry point for token validation
func (b *bridge) IsTokenValid(tokenID int, config *configSet) bool {
	analysis := b.analyzer.analysis(tokenID)

	if !analysis.HasMatches {
		return false
	}

	// Fast path: exact terminal match
	if analysis.exactMatch >= 0 {
		terminal := b.analyzer.matcher.terminals[analysis.exactMatch]
		return b.canAcceptTerminal(config, terminal.Pattern)
	}

	// General path: DP over (pos, config)
	return b.dpValidate(&analysis, config)
}

// canAcceptTerminal checks if any config can accept the terminal
func (b *bridge) canAcceptTerminal(config *configSet, pattern string) bool {
	for _, cfg := range config.configs {
		if b.canConfigAcceptTerminal(cfg, pattern) {
			return true
		}
	}
	return false
}

// canConfigAcceptTerminal checks if a single config can accept the terminal
func (b *bridge) canConfigAcceptTerminal(cfg *parserConfig, pattern string) bool {
	// Use pooled visited map to reduce allocations
	visited := getVisitedMap()
	result := b.tryAcceptTerminal(cfg.state, cfg.Stack, pattern, visited)
	putVisitedMap(visited)
	return result
}

// tryAcceptTerminal recursively tries to accept a terminal from a state
func (b *bridge) tryAcceptTerminal(state state, stack []stackSymbol, pattern string, visited map[stateStackKey]bool) bool {
	key := stateStackKey{state: state, stackSig: stackSignature(stack)}
	if visited[key] {
		return false
	}
	visited[key] = true

	stackTop := stackEmpty
	if len(stack) > 0 {
		stackTop = stack[len(stack)-1]
	}

	for _, t := range b.pda.Transitions[state] {
		// Check stack constraint
		if t.stackTop != stackEmpty && t.stackTop != stackTop {
			continue
		}

		// Can't pop more than we have
		if t.StackPop > len(stack) {
			continue
		}

		if t.Pattern == pattern {
			// Direct match
			return true
		}

		if t.Pattern == "" {
			// Epsilon transition - follow it
			newStack := make([]stackSymbol, len(stack))
			copy(newStack, stack)

			// Pop
			if t.StackPop > 0 {
				newStack = newStack[:len(newStack)-t.StackPop]
			}

			// Push
			newStack = append(newStack, t.StackPush...)

			if b.tryAcceptTerminal(t.ToState, newStack, pattern, visited) {
				return true
			}
		}
	}

	return false
}

// dpValidate runs DP for multi-terminal tokens
func (b *bridge) dpValidate(analysis *tokenAnalysis, startConfig *configSet) bool {
	// state: (pos, configSet)
	// Memoize by (pos, configSig)
	type dpKey struct {
		pos int
		sig uint64
	}
	memo := make(map[dpKey]bool)

	var dp func(pos int, config *configSet) bool
	dp = func(pos int, config *configSet) bool {
		if pos == len(analysis.Token) {
			return true // Consumed entire token
		}

		if config.isEmpty() {
			return false
		}

		key := dpKey{pos, config.signature()}
		if result, ok := memo[key]; ok {
			return result
		}

		// Try each terminal that matches at this position
		for _, match := range analysis.MatchesAtPos[pos] {
			terminal := b.analyzer.matcher.terminals[match.TerminalID]
			newConfig := b.advanceConfig(config, terminal.Pattern)
			if newConfig != nil && !newConfig.isEmpty() && dp(pos+match.Length, newConfig) {
				memo[key] = true
				return true
			}
		}

		memo[key] = false
		return false
	}

	return dp(0, startConfig)
}

// advanceConfig advances all configs that can accept the terminal
func (b *bridge) advanceConfig(config *configSet, pattern string) *configSet {
	var newConfigs []*parserConfig

	for _, cfg := range config.configs {
		advanced := b.advanceSingleConfig(cfg, pattern)
		newConfigs = append(newConfigs, advanced...)
	}

	if len(newConfigs) == 0 {
		return nil
	}

	return &configSet{configs: newConfigs}
}

// advanceSingleConfig advances a single config by accepting a terminal
func (b *bridge) advanceSingleConfig(cfg *parserConfig, pattern string) []*parserConfig {
	var results []*parserConfig
	visited := getVisitedMap()
	b.collectAdvanced(cfg.state, cfg.Stack, pattern, visited, &results)
	putVisitedMap(visited)
	return results
}

// collectAdvanced collects all configs reachable by accepting the pattern
func (b *bridge) collectAdvanced(state state, stack []stackSymbol, pattern string, visited map[stateStackKey]bool, results *[]*parserConfig) {
	key := stateStackKey{state: state, stackSig: stackSignature(stack)}
	if visited[key] {
		return
	}
	visited[key] = true

	stackTop := stackEmpty
	if len(stack) > 0 {
		stackTop = stack[len(stack)-1]
	}

	for _, t := range b.pda.Transitions[state] {
		// Check stack constraint
		if t.stackTop != stackEmpty && t.stackTop != stackTop {
			continue
		}

		// Can't pop more than we have
		if t.StackPop > len(stack) {
			continue
		}

		if t.Pattern == pattern {
			// Match! Create new config after transition
			newStack := make([]stackSymbol, len(stack))
			copy(newStack, stack)

			if t.StackPop > 0 {
				newStack = newStack[:len(newStack)-t.StackPop]
			}
			newStack = append(newStack, t.StackPush...)

			*results = append(*results, &parserConfig{
				state: t.ToState,
				Stack: newStack,
			})
		}

		if t.Pattern == "" {
			// Epsilon transition - follow it
			newStack := make([]stackSymbol, len(stack))
			copy(newStack, stack)

			if t.StackPop > 0 {
				newStack = newStack[:len(newStack)-t.StackPop]
			}
			newStack = append(newStack, t.StackPush...)

			b.collectAdvanced(t.ToState, newStack, pattern, visited, results)
		}
	}
}

// validTokens returns all token IDs that are valid from the given config
func (b *bridge) validTokens(config *configSet) []int {
	var valid []int
	for tokenID := 0; tokenID < b.analyzer.vocabSize(); tokenID++ {
		if b.IsTokenValid(tokenID, config) {
			valid = append(valid, tokenID)
		}
	}
	return valid
}

// acceptToken attempts to accept a token and returns the new config set
// Returns nil if the token is not valid from this config
func (b *bridge) acceptToken(tokenID int, config *configSet) *configSet {
	analysis := b.analyzer.analysis(tokenID)

	if !analysis.HasMatches {
		return nil
	}

	// Fast path: exact terminal match
	if analysis.exactMatch >= 0 {
		terminal := b.analyzer.matcher.terminals[analysis.exactMatch]
		newConfig := b.advanceConfig(config, terminal.Pattern)
		if newConfig != nil && !newConfig.isEmpty() {
			newConfig.normalize()
			return newConfig
		}
		return nil
	}

	// General path: DP to find final config after consuming token
	return b.dpAccept(&analysis, config)
}

// dpAccept runs DP to accept a multi-terminal token and return final config
// Returns the union of all possible end configurations (preserves nondeterminism)
func (b *bridge) dpAccept(analysis *tokenAnalysis, startConfig *configSet) *configSet {
	type dpKey struct {
		pos int
		sig uint64
	}
	// Memoize the configs reachable at each (pos, sig)
	memo := make(map[dpKey]*configSet)

	var dp func(pos int, config *configSet) *configSet
	dp = func(pos int, config *configSet) *configSet {
		if pos == len(analysis.Token) {
			return config // Consumed entire token, return final config
		}

		if config.isEmpty() {
			return nil
		}

		key := dpKey{pos, config.signature()}
		if result, ok := memo[key]; ok {
			return result
		}

		// Collect all valid result configs from all possible paths
		var allConfigs []*parserConfig

		// Try each terminal that matches at this position
		for _, match := range analysis.MatchesAtPos[pos] {
			terminal := b.analyzer.matcher.terminals[match.TerminalID]
			newConfig := b.advanceConfig(config, terminal.Pattern)
			if newConfig != nil && !newConfig.isEmpty() {
				finalConfig := dp(pos+match.Length, newConfig)
				if finalConfig != nil {
					// Collect all configs, don't return early
					allConfigs = append(allConfigs, finalConfig.configs...)
				}
			}
		}

		// Build result: nil if no valid paths, normalized configSet otherwise
		var result *configSet
		if len(allConfigs) > 0 {
			result = &configSet{configs: allConfigs}
			result.normalize() // Dedup using parserConfig.key(), sort for consistent signature
		}
		memo[key] = result // Cache normalized result
		return result
	}

	return dp(0, startConfig)
}

// isAccepting returns true if any config can reach an accepting state
func (b *bridge) isAccepting(config *configSet) bool {
	visited := getVisitedMap()
	defer putVisitedMap(visited)

	for _, cfg := range config.configs {
		// Clear visited for each config check
		for k := range visited {
			delete(visited, k)
		}
		if b.canReachAccept(cfg.state, cfg.Stack, visited) {
			return true
		}
	}
	return false
}

// canReachAccept checks if we can reach an accepting state via epsilon transitions
func (b *bridge) canReachAccept(state state, stack []stackSymbol, visited map[stateStackKey]bool) bool {
	// Check if this state is accepting with empty stack
	if b.pda.AcceptStates[state] && len(stack) == 0 {
		return true
	}

	key := stateStackKey{state: state, stackSig: stackSignature(stack)}
	if visited[key] {
		return false
	}
	visited[key] = true

	// Try epsilon transitions
	stackTop := stackEmpty
	if len(stack) > 0 {
		stackTop = stack[len(stack)-1]
	}

	for _, t := range b.pda.Transitions[state] {
		if t.Pattern != "" {
			continue // Not epsilon
		}
		if t.stackTop != stackEmpty && t.stackTop != stackTop {
			continue
		}
		if t.StackPop > len(stack) {
			continue
		}

		newStack := make([]stackSymbol, len(stack))
		copy(newStack, stack)
		if t.StackPop > 0 {
			newStack = newStack[:len(newStack)-t.StackPop]
		}
		newStack = append(newStack, t.StackPush...)

		if b.canReachAccept(t.ToState, newStack, visited) {
			return true
		}
	}

	return false
}

// validTerminals returns the valid terminal patterns from the given config
func (b *bridge) validTerminals(config *configSet) []string {
	seen := make(map[string]bool)
	var terminals []string

	visited := getVisitedMap()
	defer putVisitedMap(visited)

	for _, cfg := range config.configs {
		// Clear visited for each config
		for k := range visited {
			delete(visited, k)
		}
		b.collectValidTerminals(cfg.state, cfg.Stack, visited, seen, &terminals)
	}

	return terminals
}

// collectValidTerminals collects all reachable terminals
func (b *bridge) collectValidTerminals(state state, stack []stackSymbol, visited map[stateStackKey]bool, seen map[string]bool, terminals *[]string) {
	key := stateStackKey{state: state, stackSig: stackSignature(stack)}
	if visited[key] {
		return
	}
	visited[key] = true

	stackTop := stackEmpty
	if len(stack) > 0 {
		stackTop = stack[len(stack)-1]
	}

	for _, t := range b.pda.Transitions[state] {
		if t.stackTop != stackEmpty && t.stackTop != stackTop {
			continue
		}
		if t.StackPop > len(stack) {
			continue
		}

		if t.Pattern != "" && !seen[t.Pattern] {
			seen[t.Pattern] = true
			*terminals = append(*terminals, t.Pattern)
		}

		if t.Pattern == "" {
			newStack := make([]stackSymbol, len(stack))
			copy(newStack, stack)
			if t.StackPop > 0 {
				newStack = newStack[:len(newStack)-t.StackPop]
			}
			newStack = append(newStack, t.StackPush...)
			b.collectValidTerminals(t.ToState, newStack, visited, seen, terminals)
		}
	}
}

// validTerminalIDs returns the IDs of valid terminals from the given config
func (b *bridge) validTerminalIDs(config *configSet) []int {
	seen := make(map[int]bool)
	var terminalIDs []int

	visited := getVisitedMap()
	defer putVisitedMap(visited)

	for _, cfg := range config.configs {
		// Clear visited for each config
		for k := range visited {
			delete(visited, k)
		}
		b.collectValidTerminalIDs(cfg.state, cfg.Stack, visited, seen, &terminalIDs)
	}

	return terminalIDs
}

// collectValidTerminalIDs collects IDs of all reachable terminals
func (b *bridge) collectValidTerminalIDs(state state, stack []stackSymbol, visited map[stateStackKey]bool, seen map[int]bool, terminalIDs *[]int) {
	key := stateStackKey{state: state, stackSig: stackSignature(stack)}
	if visited[key] {
		return
	}
	visited[key] = true

	stackTop := stackEmpty
	if len(stack) > 0 {
		stackTop = stack[len(stack)-1]
	}

	for _, t := range b.pda.Transitions[state] {
		if t.stackTop != stackEmpty && t.stackTop != stackTop {
			continue
		}
		if t.StackPop > len(stack) {
			continue
		}

		if t.Pattern != "" {
			// Look up terminal ID from pattern
			if tid, ok := b.analyzer.matcher.patternToID[t.Pattern]; ok && !seen[tid] {
				seen[tid] = true
				*terminalIDs = append(*terminalIDs, tid)
			}
		}

		if t.Pattern == "" {
			newStack := make([]stackSymbol, len(stack))
			copy(newStack, stack)
			if t.StackPop > 0 {
				newStack = newStack[:len(newStack)-t.StackPop]
			}
			newStack = append(newStack, t.StackPush...)
			b.collectValidTerminalIDs(t.ToState, newStack, visited, seen, terminalIDs)
		}
	}
}
