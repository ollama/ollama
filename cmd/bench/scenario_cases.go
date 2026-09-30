package main

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/llm"
)

func cold(name, prompt string) stepSpec {
	return stepSpec{name: name, prompt: prompt, min: 0, max: coldMax}
}

// hit expects reuse of about n tokens: a full match on a prompt of n tokens.
func hit(name, prompt string, n int) stepSpec {
	return stepSpec{name: name, prompt: prompt, min: n - hitSlack, max: n}
}

// reuse expects reuse between lo and hi tokens, with slack on both ends.
func reuse(name, prompt string, lo, hi int) stepSpec {
	return stepSpec{name: name, prompt: prompt, min: lo - hitSlack, max: hi + hitSlack}
}

func runRepeat(s *scenarioRun) error {
	a, n, err := s.prompt(s.target)
	if err != nil {
		return err
	}
	if _, err := s.step(cold("cold", a)); err != nil {
		return err
	}
	_, err = s.step(hit("hit", a, n))
	return err
}

func runExtend(s *scenarioRun) error {
	a, n, err := s.prompt(s.target)
	if err != nil {
		return err
	}
	shared, err := s.count(a + tailPreamble)
	if err != nil {
		return err
	}
	if _, err := s.step(cold("cold", a)); err != nil {
		return err
	}
	_, err = s.step(reuse("extend", a+s.tail(), n, shared))
	return err
}

// runContinue appends the model's own reply before the next turn, so the match
// runs through the generated tokens the runner stored when the reply finished.
//
// The reply comes back as text, and re-tokenizing it can split it differently
// from how it was generated. Caches that cannot rewind then restore from the
// end-of-prompt snapshot, losing the whole reply. So reuse through the reply is
// required only when the runner's match covers the reply; otherwise the prompt
// is the floor, and the line's matched and cached tokens show the loss.
func runContinue(s *scenarioRun) error {
	a, n, err := s.prompt(s.target)
	if err != nil {
		return err
	}
	res, err := s.step(cold("cold", a))
	if err != nil {
		return err
	}
	conversation := a + res.content
	c, err := s.count(conversation)
	if err != nil {
		return err
	}
	res, err = s.step(reuse("continue", conversation+s.tail(), n, c))
	if err != nil {
		return err
	}
	if cached := *res.cachedPromptCount; res.stats.MatchedTokens >= c && cached < c-hitSlack {
		return fmt.Errorf("step continue: matched %d tokens through the %d-token reply but reused only %d: %w", res.stats.MatchedTokens, c, cached, errVoid)
	}
	return nil
}

func runBranch(s *scenarioRun) error {
	a, n, err := s.prompt(s.target)
	if err != nil {
		return err
	}
	shared, err := s.count(a + tailPreamble)
	if err != nil {
		return err
	}
	first := a + s.tail()
	n1, err := s.count(first)
	if err != nil {
		return err
	}
	// The prefix goes first on its own, like a system prompt: caches that
	// cannot rewind restore only from snapshots, which the runner takes near
	// each prompt's end.
	steps := []stepSpec{
		cold("cold", a),
		reuse("turn", first, n, shared),
		reuse("branch", a+s.tail(), n, shared),
		hit("return", first, n1),
	}
	for _, spec := range steps {
		if _, err := s.step(spec); err != nil {
			return err
		}
	}
	return nil
}

// runMidBranch diverges inside a long prompt, where no end-of-prompt snapshot
// exists. Caches that cannot rewind fall back to the last periodic snapshot;
// that divergence schedules a branch-point snapshot, so a second divergence at
// the same place reuses everything before it.
func runMidBranch(s *scenarioRun) error {
	target := max(s.target, 2*snapshotInterval)
	if s.ctxLen > 0 && target+1024 > s.ctxLen {
		return fmt.Errorf("context length %d is below the %d tokens this needs: %w", s.ctxLen, target, errSkip)
	}
	long, _, err := s.prompt(target)
	if err != nil {
		return err
	}
	// Cut at a problem boundary about two thirds in.
	cut := strings.LastIndex(long[:len(long)*2/3], "\n\n\n")
	if cut < 0 {
		return errors.New("prompt has no problem boundary to cut at")
	}
	prefix := long[:cut]
	m, err := s.count(prefix)
	if err != nil {
		return err
	}
	shared, err := s.count(prefix + tailPreamble)
	if err != nil {
		return err
	}
	snap := m / snapshotInterval * snapshotInterval
	steps := []stepSpec{
		cold("cold", long),
		reuse("diverge", prefix+s.tail(), snap, shared),
		reuse("rediverge", prefix+s.tail(), m, shared),
	}
	for _, spec := range steps {
		if _, err := s.step(spec); err != nil {
			return err
		}
	}
	return nil
}

func runLongGen(s *scenarioRun) error {
	a, n, err := s.prompt(s.target)
	if err != nil {
		return err
	}
	prime := cold("cold", a)
	prime.numPredict = 8
	if _, err := s.step(prime); err != nil {
		return err
	}
	long := hit("decode", a, n)
	long.numPredict = 1024
	_, err = s.step(long)
	return err
}

func runSweep(s *scenarioRun) error {
	ran := 0
	for _, size := range []int{2048, 8192, 16384, 32768} {
		if s.ctxLen > 0 && size+1024 > s.ctxLen {
			continue
		}
		a, n, err := s.prompt(size)
		if err != nil {
			return err
		}
		label := fmt.Sprintf("%dk", size/1024)
		prime := cold("cold-"+label, a)
		prime.numPredict = 8
		if _, err := s.step(prime); err != nil {
			return err
		}
		h := hit("hit-"+label, a, n)
		h.numPredict = 128
		if _, err := s.step(h); err != nil {
			return err
		}
		ran++
	}
	if ran == 0 {
		return fmt.Errorf("context length %d is too small: %w", s.ctxLen, errSkip)
	}
	return nil
}

// runEvict fills the snapshot store with distinct prompts. A prompt's
// snapshots are copied out when the next prompt moves the cache off its path,
// so each step adds one prompt's snapshots; its size is the step's growth plus
// what the step evicted. Storage must never pass the limit by more than one
// prompt. Once evictions since the first prompt outweigh everything stored
// before it plus two prompts, eviction order guarantees the first prompt is
// gone: it must miss, and the newest must hit.
func runEvict(s *scenarioRun) error {
	const size = 16384
	var first, last string
	var lastN int
	var cold0, evicted0, prevCold, prevEvicted, perPrompt int64
	limit := 400
	for i := 0; ; i++ {
		p, n, err := s.prompt(size)
		if err != nil {
			return err
		}
		spec := cold("fill", p)
		spec.numPredict = 4
		spec.quiet = true
		res, err := s.step(spec)
		if err != nil {
			return err
		}
		st := res.stats
		last, lastN = p, n
		if i == 0 {
			first = p
			cold0, evicted0 = st.ColdBytes, st.ColdEvicted
			s.emit("fill-first", res)
		} else {
			perPrompt = max(perPrompt, st.ColdBytes-prevCold+st.ColdEvicted-prevEvicted)
		}
		prevCold, prevEvicted = st.ColdBytes, st.ColdEvicted
		if i > 0 && st.ColdBytes > st.ColdLimit+perPrompt {
			return fmt.Errorf("step fill %d: snapshot storage %d bytes exceeds the %d byte limit by more than one prompt (%d): %w", i, st.ColdBytes, st.ColdLimit, perPrompt, errVoid)
		}
		if perPrompt > 0 {
			limit = min(limit, int((cold0+st.ColdLimit)/perPrompt)+8)
			if st.ColdEvicted-evicted0 >= cold0+2*perPrompt {
				s.emit("fill-full", res)
				break
			}
		}
		if i >= limit {
			return fmt.Errorf("evicted %d bytes after %d prompts of %d tokens, short of the %d stored before them plus two prompts of %d: %w", st.ColdEvicted-evicted0, i+1, size, cold0, perPrompt, errVoid)
		}
	}
	if _, err := s.step(cold("oldest", first)); err != nil {
		return err
	}
	_, err := s.step(hit("newest", last, lastN))
	return err
}

// runChurn repeats the same two conversations and checks active memory after
// each cycle stops growing once the first cycle has stored their snapshots.
func runChurn(s *scenarioRun) error {
	const cycles = 6
	x, nx, err := s.prompt(s.target)
	if err != nil {
		return err
	}
	y, ny, err := s.prompt(s.target)
	if err != nil {
		return err
	}
	var active []int64
	for c := range cycles {
		sx, sy := hit("hit-x", x, nx), hit("hit-y", y, ny)
		if c == 0 {
			sx, sy = cold("cold-x", x), cold("cold-y", y)
		}
		if _, err := s.step(sx); err != nil {
			return err
		}
		res, err := s.step(sy)
		if err != nil {
			return err
		}
		active = append(active, res.stats.ActiveBytes)
	}
	base, end := active[1], active[cycles-1]
	if tol := max(int64(64<<20), base/50); end-base > tol {
		return fmt.Errorf("active memory grew from %d to %d bytes over %d identical cycles: %w", base, end, cycles-2, errVoid)
	}
	return nil
}

// runCancel cancels a long prefill partway through and retries it. The retry
// must reuse at least one prefill chunk and must not find the whole prompt.
func runCancel(s *scenarioRun) error {
	const chunk = 2048
	target := max(s.target, 8*chunk)
	if s.ctxLen > 0 && target+1024 > s.ctxLen {
		return fmt.Errorf("context length %d is below the %d tokens this needs: %w", s.ctxLen, target, errSkip)
	}
	probe, _, err := s.prompt(target)
	if err != nil {
		return err
	}
	timing := cold("time", probe)
	timing.numPredict = 1
	timing.quiet = true
	res, err := s.step(timing)
	if err != nil {
		return err
	}

	p, n, err := s.prompt(target)
	if err != nil {
		return err
	}
	cancelAfter := max(50*time.Millisecond, res.promptEvalDuration*2/5)
	if _, err := s.step(stepSpec{name: "cancelled", prompt: p, cancel: cancelAfter}); err != nil {
		return err
	}
	_, err = s.step(stepSpec{name: "resume", prompt: p, min: chunk - hitSlack, max: n - chunk})
	return err
}

func runInterleave(s *scenarioRun) error {
	const turns = 3
	x, _, err := s.prompt(s.target)
	if err != nil {
		return err
	}
	y, _, err := s.prompt(s.target)
	if err != nil {
		return err
	}
	prev := map[string]string{"x": "", "y": ""}
	convs := map[string]string{"x": x, "y": y}
	for t := range turns {
		for _, id := range []string{"x", "y"} {
			name := fmt.Sprintf("%s%d", id, t+1)
			var spec stepSpec
			if t == 0 {
				spec = cold(name, convs[id])
			} else {
				n, err := s.count(prev[id])
				if err != nil {
					return err
				}
				shared, err := s.count(prev[id] + tailPreamble)
				if err != nil {
					return err
				}
				spec = reuse(name, convs[id], n, shared)
			}
			if _, err := s.step(spec); err != nil {
				return err
			}
			prev[id] = convs[id]
			convs[id] += s.tail()
		}
	}
	return nil
}

func runConcurrent(s *scenarioRun) error {
	x, nx, err := s.prompt(s.target)
	if err != nil {
		return err
	}
	y, ny, err := s.prompt(s.target)
	if err != nil {
		return err
	}
	pair := func(a, b stepSpec) error {
		var wg sync.WaitGroup
		errs := make([]error, 2)
		for i, spec := range []stepSpec{a, b} {
			wg.Go(func() { _, errs[i] = s.step(spec) })
		}
		wg.Wait()
		return errors.Join(errs...)
	}
	if err := pair(cold("cold-x", x), cold("cold-y", y)); err != nil {
		return err
	}
	return pair(hit("hit-x", x, nx), hit("hit-y", y, ny))
}

// runShort sends prompts of a handful of tokens, where the seed back-off and
// the end-of-prompt snapshot offset run into the start of the prompt. Only the
// universal checks apply: never the whole prompt, exact generation.
func runShort(s *scenarioRun) error {
	for _, letters := range []int{1, 3, 5, 8} {
		p := promptNonce(letters)
		n, err := s.count(p)
		if err != nil {
			return err
		}
		for _, name := range []string{"cold", "hit"} {
			spec := stepSpec{name: fmt.Sprintf("%s-%d", name, n), prompt: p, min: 0, max: n}
			if _, err := s.step(spec); err != nil {
				return err
			}
		}
	}
	return nil
}

func runMedia(s *scenarioRun) error {
	image := func() []llm.MediaData {
		return []llm.MediaData{{ID: 0, Kind: llm.MediaKindImage, Data: scenarioPNG(s.variation())}}
	}
	body := func() string {
		return nonceHeader(promptNonce(nonceLetters)) + codePromptBody(max(smallestCodePromptWords(), s.target/8), s.variation())
	}

	// Image first, then text: the image is part of the reused prefix.
	pre := body() + "\n\n"
	before, err := s.count(pre)
	if err != nil {
		return err
	}
	a := pre + "[img-0]\n\n" + codePromptBody(max(smallestCodePromptWords(), s.target/8), s.variation())
	media := image()
	res, err := s.step(stepSpec{name: "cold", prompt: a, media: media, min: 0, max: coldMax})
	if err != nil {
		if unsupportedMedia(err) {
			return fmt.Errorf("model has no image input: %w", errSkip)
		}
		return err
	}
	n := res.promptEvalCount
	if _, err := s.step(stepSpec{name: "hit", prompt: a, media: media, min: n - hitSlack, max: n}); err != nil {
		return err
	}
	if _, err := s.step(stepSpec{name: "extend", prompt: a + s.tail(), media: media, min: n - hitSlack, max: n + 2*hitSlack}); err != nil {
		return err
	}

	// A different image in the same place must not match the first one.
	if _, err := s.step(stepSpec{name: "other-image", prompt: a, media: image(), min: 0, max: before + hitSlack}); err != nil {
		return err
	}

	// Image last: a full match ends inside or right after the image. The runner
	// backs off to the image's start when the model attends to images
	// bidirectionally, and caches that cannot rewind then restore from the
	// nearest snapshot before it, which today is only the shared header: the
	// end-of-prompt snapshot falls inside the image. Only the universal checks
	// apply; the line reports the reuse.
	text := body() + "\n\n"
	last := image()
	if _, err := s.step(stepSpec{name: "cold-last", prompt: text + "[img-0]", media: last, min: 0, max: coldMax}); err != nil {
		return err
	}
	if _, err := s.step(stepSpec{name: "hit-last", prompt: text + "[img-0]", media: last, min: 0, max: 1 << 30}); err != nil {
		return err
	}

	// Audio, where the model takes it: the same checks as the image prefix.
	audio := []llm.MediaData{{ID: 0, Kind: llm.MediaKindAudio, Data: scenarioWAV(s.variation())}}
	sound := body() + "\n\n[img-0]\n\n" + codePromptBody(max(smallestCodePromptWords(), s.target/8), s.variation())
	res, err = s.step(stepSpec{name: "cold-audio", prompt: sound, media: audio, min: 0, max: coldMax})
	if err != nil {
		if unsupportedMedia(err) {
			return nil
		}
		return err
	}
	na := res.promptEvalCount
	_, err = s.step(stepSpec{name: "hit-audio", prompt: sound, media: audio, min: na - hitSlack, max: na})
	return err
}

// runModes switches request modes over one conversation. A logprobs request
// parks speculative drafting while the draft cache is kept level with the
// target; structured output decodes through the grammar path. Each later
// request must still reuse the cache the earlier one left.
func runModes(s *scenarioRun) error {
	a, n, err := s.prompt(s.target)
	if err != nil {
		return err
	}
	shared, err := s.count(a + tailPreamble)
	if err != nil {
		return err
	}
	parked := cold("cold-logprobs", a)
	parked.logprobs = true
	res, err := s.step(parked)
	if err != nil {
		return err
	}
	if res.stats.DraftTokens != 0 {
		return fmt.Errorf("step cold-logprobs: drafted %d tokens with logprobs requested: %w", res.stats.DraftTokens, errVoid)
	}
	if _, err := s.step(hit("hit", a, n)); err != nil {
		return err
	}
	ab := a + s.tail()
	structured := reuse("extend-json", ab, n, shared)
	structured.format = `{"type":"structural_tag","format":{"type":"json_schema","json_schema":{"type":"object"}}}`
	if _, err := s.step(structured); err != nil {
		return err
	}
	nb, err := s.count(ab)
	if err != nil {
		return err
	}
	_, err = s.step(hit("hit-after-json", ab, nb))
	return err
}

// unsupportedMedia reports whether the runner refused media the model cannot take.
func unsupportedMedia(err error) bool {
	return strings.Contains(err.Error(), "does not support") || strings.Contains(err.Error(), "is unavailable")
}
