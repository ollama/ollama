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

// runCancelledPrefill mirrors a prefill cancelled by a client timeout at
// (roughly) cancelAt prompt tokens: begin, schedule, feed whole chunks until
// cancelAt with the drafter's pairs lagging one token, settle the drafter with
// the next prompt token, then close, which attaches the crossed captures.
func (e *hybridEnv) runCancelledPrefill(t *testing.T, inputs []int32, chunk, cancelAt int) {
	t.Helper()
	pc := e.pc
	session := pc.begin(inputs, nil)
	session.schedulePrefillSnapshots(periodic(len(inputs)))

	pos := pc.minCacheOffset()
	for pos < cancelAt && pos < len(inputs)-1 {
		n := min(chunk, len(inputs)-1-pos)
		for _, c := range pc.caches {
			if c == cache.Cache(e.draft) {
				continue
			}
			if fc, ok := c.(feedableCache); ok {
				fc.feed(inputs[pos : pos+n])
			}
		}
		if d := e.draft.Offset(); d < pos+n-1 {
			e.draft.feed(inputs[d : pos+n-1])
		}
		pos += n
	}

	if d := e.draft.Offset(); d < pos {
		e.draft.feed(inputs[d:pos])
	}
	session.close()
}

// TestScenarioCancelledPrefills covers the cancellation invariants: a retry
// of a cancelled prefill resumes exactly where the previous attempt stopped,
// and the captures the cancelled attempt crossed become restore points.
func TestScenarioCancelledPrefills(t *testing.T) {
	logs := captureWarns(t)
	e := newHybridEnv()
	ts := &tokenStream{}
	begins := &beginLog{t: t, logs: logs}

	prompt := ts.fresh(47)

	// Cancelled after two chunks of 9. The prefill crossed the captures
	// scheduled at interval and 2*interval — key offsets 7 and 15 after the
	// draft look-ahead shift — and close attached them.
	e.runCancelledPrefill(t, prompt, 9, 2*9)
	begins.next()

	// A session diverging inside the cancelled span restores to the deepest
	// crossed capture: 17 keys match, and the entry at 15 is the closest
	// restore point below.
	probe := slices.Concat(slices.Clone(prompt[:18]), ts.fresh(10))
	pr := e.pc.begin(probe, nil)
	if _, m, c, _ := begins.next(); m != 17 || c != 15 {
		t.Errorf("probe into cancelled span: matched=%d cached=%d, want 17/15", m, c)
	}
	pr.close()

	// The retry resumes exactly at the 18 tokens the first attempt recorded,
	// and is cancelled again two chunks deeper.
	e.runCancelledPrefill(t, prompt, 9, 2*9+2*9)
	if _, m, c, _ := begins.next(); m != 18 || c != 18 {
		t.Errorf("first retry: matched=%d cached=%d, want 18/18", m, c)
	}

	// The final retry resumes at 36 and completes.
	e.runRequest(t, prompt, ts.fresh(6))
	if _, m, c, _ := begins.next(); m != 36 || c != 36 {
		t.Errorf("second retry: matched=%d cached=%d, want 36/36", m, c)
	}

	if out := logs(); strings.Contains(out, "failed to restore cache") {
		t.Errorf("freeAll warn fired:\n%s", out)
	}
	checkSnapshotCoverage(t, e.pc, e.kvLayers())
}

// TestScenarioDivergentCancels interleaves cancelled prefills that diverge
// from a conversation mid-history with completed requests, growing branch
// points above and below the conversation's capture offsets, then re-requests
// the full conversation, which must restore to within the capture cadence.
func TestScenarioDivergentCancels(t *testing.T) {
	logs := captureWarns(t)
	e := newHybridEnv()
	ts := &tokenStream{}
	begins := &beginLog{t: t, logs: logs}

	prompt := ts.fresh(24)
	gen := ts.fresh(8)
	e.runRequest(t, prompt, gen)
	stream := slices.Concat(prompt, gen)
	begins.next()

	// A cancelled prefill diverging between the second and third captures.
	e.runCancelledPrefill(t, slices.Concat(slices.Clone(stream[:2*interval+2]), ts.fresh(20)), 5, 12)
	begins.next()

	// A completed turn extends the conversation.
	p2 := slices.Concat(stream, ts.fresh(10))
	g2 := ts.fresh(4)
	e.runRequest(t, p2, g2)
	stream = slices.Concat(p2, g2)
	begins.next()

	// A cancelled prefill diverging below the first capture.
	e.runCancelledPrefill(t, slices.Concat(slices.Clone(stream[:interval-2]), ts.fresh(16)), 5, 14)
	begins.next()

	// Re-requesting the full conversation restores to within the cadence of
	// the last completed turn.
	e.runRequest(t, slices.Concat(stream, ts.fresh(12)), ts.fresh(5))
	if _, m, c, _ := begins.next(); c < m-e.restoreBound(len(g2), 0) {
		t.Errorf("re-request: cached=%d fell more than %d below matched=%d", c, e.restoreBound(len(g2), 0), m)
	}

	if out := logs(); strings.Contains(out, "failed to restore cache") {
		t.Errorf("freeAll warn fired:\n%s", out)
	}
	checkSnapshotCoverage(t, e.pc, e.kvLayers())
}
