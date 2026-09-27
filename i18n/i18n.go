// Package i18n localizes ollama's command line interface into Simplified
// Chinese while keeping every other locale in English.
//
// English source strings double as translation keys (gettext style), so the
// English output path stays byte-for-byte identical to the untranslated code
// and any missing translation falls back to English automatically.
//
// Locale resolution order:
//
//  1. OLLAMA_LANG (any zh* locale selects Chinese, anything else English)
//  2. when running under `go test`: English (keeps test assertions stable)
//  3. LC_ALL, LC_MESSAGES, LANGUAGE, LANG — the first variable that carries
//     real locale information (C/POSIX are treated as "no opinion")
//  4. the system language, consulted only when the environment carries no
//     locale information at all (macOS AppleLanguages)
//  5. English
package i18n

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// Locale reports the active locale: "zh" for Simplified Chinese, "en" for
// English. It is resolved on every call so tests can override OLLAMA_LANG;
// the system language probe is memoized.
func Locale() string {
	if v := os.Getenv("OLLAMA_LANG"); v != "" {
		return localeOf(v)
	}
	if runningUnderTest() {
		return "en"
	}
	for _, key := range []string{"LC_ALL", "LC_MESSAGES", "LANGUAGE", "LANG"} {
		if v := os.Getenv(key); v != "" && !posixLocale(v) {
			return localeOf(v)
		}
	}
	return systemLocale()
}

// T returns the Simplified Chinese translation of the English string s. When
// the active locale is not Chinese, or no translation exists, s is returned
// unchanged.
func T(s string) string {
	if Locale() != "zh" {
		return s
	}
	if v, ok := messages()[s]; ok {
		return v
	}
	return s
}

// Status translates a progress status line for display. Exact catalog matches
// are tried first; the progress prefix rules then cover statuses built
// dynamically, such as "pulling <digest>". Server-provided statuses stay
// English on the wire and are only translated at the display site.
func Status(s string) string {
	if Locale() != "zh" {
		return s
	}
	if v, ok := messages()[s]; ok {
		return v
	}
	for _, p := range statusPrefixes {
		if rest, found := strings.CutPrefix(s, p[0]); found {
			return p[1] + rest
		}
	}
	return s
}

// Err returns err with well-known cobra, pflag and Modelfile parser messages
// rewritten in Chinese when the locale is Chinese. Errors that match no
// template are returned unchanged, so Err is safe to apply to every error the
// CLI prints: already-translated and untranslated server/API messages pass
// through untouched.
func Err(err error) error {
	if err == nil || Locale() != "zh" {
		return err
	}
	return errors.New(translateMessage(err.Error()))
}

// LocalizeUsageTemplate rewrites the English section labels embedded in a
// cobra usage template. In English every replacement is the identity, so the
// template comes back unchanged.
func LocalizeUsageTemplate(t string) string {
	repl := strings.NewReplacer(
		"Usage:", T("Usage:"),
		"Aliases:", T("Aliases:"),
		"Examples:", T("Examples:"),
		"Available Commands:", T("Available Commands:"),
		"Additional Commands:", T("Additional Commands:"),
		"Flags:", T("Flags:"),
		"Global Flags:", T("Global Flags:"),
		"Additional help topics:", T("Additional help topics:"),
		`Use "{{.CommandPath}} [command] --help" for more information about a command.`,
		T(`Use "{{.CommandPath}} [command] --help" for more information about a command.`),
	)
	return repl.Replace(t)
}

// HelpForUsage returns the translated --help flag usage for a command name.
func HelpForUsage(name string) string {
	if name == "" {
		name = "this command"
	}
	return fmt.Sprintf(T("help for %s"), name)
}

// localeOf maps an environment-style locale value onto "zh" or "en".
func localeOf(v string) string {
	if isZhLocale(v) {
		return "zh"
	}
	return "en"
}

// posixLocale reports whether v is a POSIX placeholder locale (C, POSIX or a
// UTF-8 variant) that carries no language preference.
func posixLocale(v string) bool {
	switch langTag(v) {
	case "", "c", "posix":
		return true
	}
	return false
}

// langTag lowercases v and reduces it to its language tag: everything before
// the first '.' or '@', with '_' folded to '-'.
func langTag(v string) string {
	if i := strings.IndexAny(v, ".@"); i >= 0 {
		v = v[:i]
	}
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(v)), "_", "-")
}

// isZhLocale reports whether v selects Chinese. Traditional Chinese locales
// (zh_TW, zh_HK, zh_MO, zh-Hant) have no Simplified catalog and fall back to
// English.
func isZhLocale(v string) bool {
	lang, rest, _ := strings.Cut(langTag(v), "-")
	if lang != "zh" {
		return false
	}
	for _, part := range strings.Split(rest, "-") {
		switch part {
		case "", "cn", "sg", "hans":
			continue
		case "tw", "hk", "mo", "hant":
			return false
		}
	}
	return true
}

// testBinaryName reports whether os.Args[0] looks like a `go test` binary.
// Evaluated once at init, when the test.v flag is not yet registered.
var testBinaryName = func() bool {
	base := filepath.Base(os.Args[0])
	return strings.HasSuffix(base, ".test") || strings.HasSuffix(base, ".test.exe")
}()

// runningUnderTest reports whether the current process is a `go test` binary.
// Both signals are used: the binary name is visible during package init, while
// the test flag is only registered once testing.MainStart has run.
func runningUnderTest() bool {
	if testBinaryName {
		return true
	}
	return flag.Lookup("test.v") != nil
}

var systemLocaleOnce = sync.OnceValue(func() string {
	if runtime.GOOS != "darwin" {
		return "en"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "defaults", "read", "-g", "AppleLanguages").Output()
	if err != nil {
		return "en"
	}
	langs := string(out)
	if start := strings.IndexByte(langs, '"'); start >= 0 {
		if end := strings.IndexByte(langs[start+1:], '"'); end >= 0 {
			if isZhLocale(langs[start+1 : start+1+end]) {
				return "zh"
			}
		}
	}
	return "en"
})

// systemLocale consults the operating system's preferred language. It is only
// reached when the environment carries no locale information.
func systemLocale() string {
	return systemLocaleOnce()
}

// messages lazily merges every catalog file into a single lookup table.
var messages = sync.OnceValue(func() map[string]string {
	merged := make(map[string]string)
	for _, src := range []map[string]string{
		zhHelp, zhCmd, zhChat, zhTui,
		zhLaunch, zhLaunch2, zhCreate,
		zhProgress, zhMisc,
	} {
		for k, v := range src {
			merged[k] = v
		}
	}
	return merged
})

// statusPrefixes holds prefix rules for dynamically built progress statuses
// (e.g. "pulling <digest>"). Exact catalog matches are tried first, so static
// statuses take precedence over these rules.
var statusPrefixes = [][2]string{
	{"pulling ", "拉取 "},
	{"pushing ", "推送 "},
	{"using autodetected template ", "使用自动检测的模板 "},
	{"couldn't remove unused layers: ", "移除未使用的层失败："},
}
