package mlxrunner

import (
	"bytes"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/cache"
)

// Scenario tests drive prefixCache through multi-request timelines on the
// production cache layout of a hybrid model with MTP speculation: interleaved
// recurrent (GDN) and rewindable (attention KV) target layers plus a draft KV
// cache with one token of look-ahead. They verify the cache lifecycle
// invariants: a request extending the active conversation resumes exactly, a
// request diverging below the frontier restores to within the capture cadence
// of its match, and every stored snapshot spans exactly its node's edge.

// The capture cadence of the scenario prompts: a snapshot every interval
// tokens plus one preThinking tokens before each prompt's end.
const (
	interval    = 8
	preThinking = 4
)

// restoreBound is how far a restore may trail its match. Captures land every
// interval tokens and preThinking tokens before a prompt's end, shifted back
// by the draft look-ahead, and the tokens generated or dropped since the last
// prompt have no user capture of their own.
func (e *hybridEnv) restoreBound(prevGen, dropped int) int {
	return interval + preThinking + e.pc.draftLookahead + prevGen + dropped
}

// captureWarns redirects slog to a buffer for the duration of the test and
// returns a func that reports captured output.
func captureWarns(t *testing.T) func() string {
	t.Helper()
	var buf bytes.Buffer
	old := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(old) })
	return func() string { return buf.String() }
}

// hybridEnv mirrors the production layout for a hybrid model with MTP:
// interleaved recurrent (GDN) and rewindable (attention KV) target layers,
// then the draft KV cache last, with draftLookahead=1.
type hybridEnv struct {
	pc    *prefixCache
	draft *fakeRewindableCache
}

func newHybridEnv() *hybridEnv {
	tr := &snapshotTracker{}
	targets := []cache.Cache{
		&fakeRecurrentCache{tracker: tr},
		&fakeRewindableCache{tracker: tr},
		&fakeRecurrentCache{tracker: tr},
		&fakeRewindableCache{tracker: tr},
	}
	draft := &fakeRewindableCache{tracker: tr}
	pc := &prefixCache{caches: append(targets, draft)}
	pc.draftLookahead = 1
	return &hybridEnv{pc: pc, draft: draft}
}

// kvLayers returns the cache indices whose snapshots are position-sliceable.
func (e *hybridEnv) kvLayers() []int {
	var layers []int
	for i, c := range e.pc.caches {
		if _, ok := c.(*fakeRewindableCache); ok {
			layers = append(layers, i)
		}
	}
	return layers
}

// tokenStream hands out distinct tokens so prompts never collide by accident.
type tokenStream struct{ next int32 }

func (ts *tokenStream) fresh(n int) []int32 {
	out := make([]int32, n)
	for i := range out {
		ts.next++
		out[i] = ts.next
	}
	return out
}

// periodic returns the scenario capture schedule for a prompt: every interval
// tokens, plus one preThinking tokens before the end.
func periodic(promptLen int) []int {
	var offs []int
	for o := interval; o < promptLen; o += interval {
		offs = append(offs, o)
	}
	if end := promptLen - preThinking; end > 0 {
		offs = append(offs, end)
	}
	return offs
}

// parseBeginLine extracts the counters from a begin log line. ok is false for
// any other line.
func parseBeginLine(line string) (total, matched, cached, left int, ok bool) {
	if !strings.Contains(line, "cache hit") && !strings.Contains(line, "cache miss") {
		return 0, 0, 0, 0, false
	}
	for _, f := range strings.Fields(line) {
		fmt.Sscanf(f, "total=%d", &total)
		fmt.Sscanf(f, "matched=%d", &matched)
		fmt.Sscanf(f, "cached=%d", &cached)
		fmt.Sscanf(f, "left=%d", &left)
	}
	return total, matched, cached, left, true
}

// beginLog scans the captured log for begin lines, one call per request.
type beginLog struct {
	t    *testing.T
	logs func() string
	mark int
}

// next returns the counters of the begin line logged since the previous call.
func (b *beginLog) next() (total, matched, cached, left int) {
	b.t.Helper()
	out := b.logs()
	seg := out[b.mark:]
	b.mark = len(out)
	for _, l := range strings.Split(seg, "\n") {
		if total, matched, cached, left, ok := parseBeginLine(l); ok {
			return total, matched, cached, left
		}
	}
	b.t.Fatalf("no begin line in:\n%s", seg)
	return 0, 0, 0, 0
}

// runRequest mirrors the production pipeline for a successful request:
// begin -> schedule periodic snapshots -> prefill all but the last prompt
// token -> attach -> decode generated tokens (all caches level, last token
// unforwarded) -> close.
func (e *hybridEnv) runRequest(t *testing.T, inputs, generated []int32) *cacheSession {
	t.Helper()
	pc := e.pc
	session := pc.begin(inputs, nil)

	session.schedulePrefillSnapshots(periodic(len(inputs)))
	base := pc.minCacheOffset()
	seed := len(inputs) - 1
	if base < seed {
		feedAll(pc.caches, inputs[base:seed])
	}
	session.attachPrefillSnapshots()

	if len(generated) > 0 {
		session.outputs = generated
		feedAll(pc.caches, inputs[seed:])
		feedAll(pc.caches, generated[:len(generated)-1])
	}
	session.close()
	return session
}

// checkSnapshotCoverage asserts the snapshot invariant over the trie: every
// node carries a snapshot for each position-sliceable (KV) layer, spanning
// exactly its edge.
func checkSnapshotCoverage(t *testing.T, pc *prefixCache, kvLayers []int) {
	t.Helper()
	walkNodes(pc.root, func(n *trieNode) bool {
		if n.endOffset == n.startOffset() { // root has an empty edge
			return true
		}
		for _, layer := range kvLayers {
			var snap *fakeSnapshot
			if layer < len(n.snapshots) && n.snapshots[layer] != nil {
				snap = n.snapshots[layer].(*fakeSnapshot)
			}
			if snap == nil {
				t.Errorf("node [%d,%d) layer %d has no snapshot", n.startOffset(), n.endOffset, layer)
				continue
			}
			if snap.from != n.startOffset() || snap.to != n.endOffset {
				t.Errorf("node [%d,%d) layer %d snapshot [%d,%d) does not span the edge", n.startOffset(), n.endOffset, layer, snap.from, snap.to)
			}
		}
		return true
	})
}

// TestScenarioConversationTurns advances a conversation turn by turn. A turn
// keeping the whole previous response extends the frontier and must resume
// exactly; a turn dropping part of it diverges below the frontier and must
// restore to within the capture cadence of its match; and a switch to an
// unrelated session must not cost the conversation more than that same bound
// when the next turn returns to it.
func TestScenarioConversationTurns(t *testing.T) {
	logs := captureWarns(t)
	e := newHybridEnv()
	ts := &tokenStream{}
	begins := &beginLog{t: t, logs: logs}

	type turn struct {
		keepGen int // previous generation kept in the prompt (-1 = all)
		newTail int
		genLen  int
		away    bool // an unrelated session runs before this turn
	}
	turns := []turn{
		{keepGen: 3, newTail: 9, genLen: 6},
		{keepGen: -1, newTail: 4, genLen: 5},
		{keepGen: 4, newTail: 6, genLen: 7},
		{keepGen: -1, newTail: 5, genLen: 6},
		{keepGen: 5, newTail: 8, genLen: 8},
		{keepGen: 4, newTail: 7, genLen: 6, away: true},
		{keepGen: -1, newTail: 6, genLen: 7},
		{keepGen: 3, newTail: 8, genLen: 6},
		{keepGen: -1, newTail: 5, genLen: 5},
		{keepGen: 4, newTail: 9, genLen: 6},
	}

	prompt := ts.fresh(50)
	gen := ts.fresh(6)
	e.runRequest(t, prompt, gen)
	begins.next()
	stream := prompt // committed conversation, minus the latest response
	lastGen := gen   // latest response

	for i, tn := range turns {
		if tn.away {
			e.runRequest(t, ts.fresh(24), ts.fresh(4))
			begins.next()
		}

		dropped := 0
		var p []int32
		if tn.keepGen < 0 {
			p = slices.Concat(stream, lastGen, ts.fresh(tn.newTail))
		} else {
			dropped = len(lastGen) - tn.keepGen
			p = slices.Concat(stream, slices.Clone(lastGen[:tn.keepGen]), ts.fresh(tn.newTail))
		}
		bound := e.restoreBound(len(lastGen), dropped)
		g := ts.fresh(tn.genLen)
		e.runRequest(t, p, g)

		_, matched, cached, _ := begins.next()
		if dropped == 0 && !tn.away && cached != matched {
			t.Errorf("turn %d: extension resumed cached=%d != matched=%d", i, cached, matched)
		}
		if cached < matched-bound {
			t.Errorf("turn %d: cached=%d fell more than %d below matched=%d", i, cached, bound, matched)
		}

		stream = p
		lastGen = g
	}

	if out := logs(); strings.Contains(out, "failed to restore cache") {
		t.Errorf("freeAll warn fired:\n%s", out)
	}
	checkSnapshotCoverage(t, e.pc, e.kvLayers())
}
