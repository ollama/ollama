// Package forkversion orders this fork's release tags and picks the next one.
// It has no cgo, so the release script can run it on Linux and on macOS.
package forkversion

import (
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// gitDescribeSuffix matches what `git describe --long --dirty` appends to the
// tag: "-<commits>-g<sha>" and an optional "-dirty".
var gitDescribeSuffix = regexp.MustCompile(`-[0-9]+-g[0-9a-f]+(-dirty)?$`)

// BuildTag returns the release tag (without "v") a build was made from,
// e.g. "0.35.1-b11325-glm5next-unload-9-g6c178a3-dirty" ->
// "0.35.1-b11325-glm5next-unload".
func BuildTag(version string) string {
	version = strings.TrimPrefix(strings.TrimSpace(version), "v")
	if loc := gitDescribeSuffix.FindStringIndex(version); loc != nil && loc[0] > 0 {
		return version[:loc[0]]
	}
	return strings.TrimSuffix(version, "-dirty")
}

var (
	forkSemver   = regexp.MustCompile(`^([0-9]+)\.([0-9]+)\.([0-9]+)`)
	forkLlamaRev = regexp.MustCompile(`(?:^|-)b([0-9]+)(?:-|$)`)
	trailingNum  = regexp.MustCompile(`^(.*)\.([0-9]+)$`)
	ds41Tag      = regexp.MustCompile(`^v[0-9]+\.[0-9]+\.[0-9]+-b[0-9]+-ds41(?:\.[0-9]+)?$`)
	ds41Bump     = regexp.MustCompile(`^(.*-ds41)\.([0-9]+)$`)
)

// Compare orders fork tags like "0.35.1-b11325-glm5next" by their X.Y.Z
// version, then by the llama.cpp bNNNNN revision when both have one, then by
// a shared trailing ".<digits>" (so ds41.4 is newer than ds41.3).
// ok is false when either tag has no X.Y.Z version.
func Compare(a, b string) (cmp int, ok bool) {
	a = strings.TrimPrefix(strings.TrimSpace(a), "v")
	b = strings.TrimPrefix(strings.TrimSpace(b), "v")
	am, bm := forkSemver.FindStringSubmatch(a), forkSemver.FindStringSubmatch(b)
	if am == nil || bm == nil {
		return 0, false
	}
	for i := 1; i <= 3; i++ {
		if c := compareNumeric(am[i], bm[i]); c != 0 {
			return c, true
		}
	}
	ar, br := forkLlamaRev.FindStringSubmatch(a), forkLlamaRev.FindStringSubmatch(b)
	if ar != nil && br != nil {
		if c := compareNumeric(ar[1], br[1]); c != 0 {
			return c, true
		}
	} else if ar != nil || br != nil {
		return 0, true
	}
	if c, ok := compareTrailing(a, b); ok && c != 0 {
		return c, true
	}
	return 0, true
}

func compareTrailing(a, b string) (int, bool) {
	am, bm := trailingNum.FindStringSubmatch(a), trailingNum.FindStringSubmatch(b)
	if am != nil && bm != nil && am[1] == bm[1] {
		return compareNumeric(am[2], bm[2]), true
	}
	// vX-ds41 is the release before vX-ds41.1.
	if am != nil && bm == nil && am[1] == b {
		return 1, true
	}
	if bm != nil && am == nil && bm[1] == a {
		return -1, true
	}
	return 0, false
}

func compareNumeric(a, b string) int {
	x, _ := strconv.ParseUint(a, 10, 64)
	y, _ := strconv.ParseUint(b, 10, 64)
	switch {
	case x < y:
		return -1
	case x > y:
		return 1
	}
	return 0
}

// IsUpdate reports whether the fork release tag should be offered to a build
// reporting version current (version.Version, which is `git describe` output
// such as "0.35.1-b11325-glm5next-unload-9-g6c178a3").
//
//   - A build made from the release's tag, or from commits after it, never
//     gets that release offered again.
//   - A release with a lower X.Y.Z, bNNNNN, or trailing .N than the build is
//     never offered.
//   - A release with a higher X.Y.Z, bNNNNN, or trailing .N (ds41.4 after
//     ds41.3) is offered even when it was published before this build.
//   - Otherwise (same numbers, different suffix such as "-glm5next" vs
//     "-glm5next-unload"), the release is offered only if it was published
//     after this build was installed, so a freshly built tag that is not
//     published yet is not "updated" back to the previous release.
func IsUpdate(tag, current string, publishedAt, installedAt time.Time) bool {
	tag = strings.TrimPrefix(strings.TrimSpace(tag), "v")
	if tag == "" {
		return false
	}
	base := BuildTag(current)
	if base == tag {
		return false
	}
	if c, ok := Compare(tag, base); ok && c != 0 {
		return c > 0
	}
	if !publishedAt.IsZero() && !installedAt.IsZero() && !publishedAt.After(installedAt) {
		return false
	}
	return true
}

// Latest returns the newest vX.Y.Z-bNNNNN-ds41[.N] tag in tags.
func Latest(tags []string) (string, error) {
	var best string
	for _, raw := range tags {
		tag := strings.TrimSpace(raw)
		if !ds41Tag.MatchString(tag) {
			continue
		}
		if best == "" {
			best = tag
			continue
		}
		c, ok := Compare(tag, best)
		if ok && c > 0 {
			best = tag
		}
	}
	if best == "" {
		return "", errors.New("no vX.Y.Z-bNNNNN-ds41 tag found")
	}
	return best, nil
}

// Next returns the tag to release for spec. spec is "patch" or an explicit
// tag in the ds41 scheme. An explicit tag must be strictly newer than Latest.
func Next(spec string, tags []string) (string, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return "", errors.New("version is required (patch or vX.Y.Z-bNNNNN-ds41.N)")
	}
	latest, latestErr := Latest(tags)
	if spec == "patch" {
		if latestErr != nil {
			return "", latestErr
		}
		return bumpPatch(latest)
	}
	tag := spec
	if !strings.HasPrefix(tag, "v") {
		tag = "v" + tag
	}
	if !ds41Tag.MatchString(tag) {
		return "", fmt.Errorf("version must be patch or vX.Y.Z-bNNNNN-ds41.N, got %q", spec)
	}
	if latestErr == nil {
		c, ok := Compare(tag, latest)
		if !ok || c <= 0 {
			return "", fmt.Errorf("%s is not newer than %s", tag, latest)
		}
	}
	return tag, nil
}

func bumpPatch(tag string) (string, error) {
	base := strings.TrimPrefix(tag, "v")
	if m := ds41Bump.FindStringSubmatch(base); m != nil {
		n, err := strconv.ParseUint(m[2], 10, 64)
		if err != nil {
			return "", err
		}
		return "v" + m[1] + "." + strconv.FormatUint(n+1, 10), nil
	}
	if strings.HasSuffix(base, "-ds41") {
		return "v" + base + ".1", nil
	}
	return "", fmt.Errorf("cannot bump %s", tag)
}
