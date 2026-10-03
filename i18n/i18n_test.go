package i18n

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/ollama/ollama/envconfig"
)

func TestLocaleDetection(t *testing.T) {
	cases := []struct {
		env  string
		want string
	}{
		{"zh", "zh"}, {"zh-CN", "zh"}, {"zh_CN", "zh"}, {"zh_CN.UTF-8", "zh"},
		{"zh-Hans", "zh"}, {"zh-Hans-JP", "zh"}, {"zh-SG", "zh"},
		{"zh-TW", "en"}, {"zh_TW", "en"}, {"zh-Hant", "en"}, {"zh-HK", "en"},
		{"en", "en"}, {"en_US.UTF-8", "en"}, {"ja_JP.UTF-8", "en"},
		{"C", "en"}, {"POSIX", "en"}, {"C.UTF-8", "en"},
	}
	for _, c := range cases {
		t.Setenv("OLLAMA_LANG", c.env)
		if got := Locale(); got != c.want {
			t.Errorf("OLLAMA_LANG=%q: Locale() = %q, want %q", c.env, got, c.want)
		}
	}
}

func TestLocalePrefersOverride(t *testing.T) {
	t.Setenv("OLLAMA_LANG", "zh-CN")
	t.Setenv("LC_ALL", "en_US.UTF-8")
	if got := Locale(); got != "zh" {
		t.Errorf("OLLAMA_LANG must override LC_ALL, got %q", got)
	}
	t.Setenv("OLLAMA_LANG", "en")
	t.Setenv("LC_ALL", "zh_CN.UTF-8")
	if got := Locale(); got != "en" {
		t.Errorf("OLLAMA_LANG=en must force English, got %q", got)
	}
}

// TestEnglishFallback verifies T is the identity in English and for unknown
// keys in Chinese.
func TestEnglishFallback(t *testing.T) {
	t.Setenv("OLLAMA_LANG", "en")
	if got := T("Run a model"); got != "Run a model" {
		t.Errorf("English T() = %q, want identity", got)
	}
	t.Setenv("OLLAMA_LANG", "zh-CN")
	if got := T("no such key anywhere"); got != "no such key anywhere" {
		t.Errorf("unknown key fallback = %q, want identity", got)
	}
}

func TestStatusTranslation(t *testing.T) {
	t.Setenv("OLLAMA_LANG", "zh-CN")
	// Exact catalog keys win over prefix rules.
	if got := Status("pulling manifest"); got == "pulling manifest" {
		t.Errorf("Status exact key not translated: %q", got)
	}
	// Dynamic statuses fall through to prefix rules.
	if got := Status("pulling 4a5b6c7d8e9f"); !strings.HasPrefix(got, "拉取 ") {
		t.Errorf("Status prefix rule = %q, want prefix 拉取 ", got)
	}
	// Interpolated statuses match display-side templates (shared create/manifest
	// packages stay English at the source).
	for in, want := range map[string]string{
		"importing model.safetensors (123 tensors, converting fp8 to mxfp8)": "导入 model.safetensors（123 个张量，将 fp8 转换为 mxfp8）",
		"importing config xform.json":                                        "导入配置 xform.json",
		"creating new layer sha256:abc":                                      "创建新层 sha256:abc",
		"successfully imported m.safetensors with 12 layers":                 "成功导入 m.safetensors，共 12 层",
		"couldn't remove unused layers: open /x: permission denied":          "无法移除未使用层：open /x: permission denied",
	} {
		if got := Status(in); got != want {
			t.Errorf("Status(%q) = %q, want %q", in, got, want)
		}
	}
	t.Setenv("OLLAMA_LANG", "en")
	if got := Status("pulling manifest"); got != "pulling manifest" {
		t.Errorf("English Status() = %q, want identity", got)
	}
}

func TestErrTemplatesTranslated(t *testing.T) {
	for _, tpl := range errTemplates {
		if _, ok := messages()[tpl]; !ok {
			t.Errorf("errTemplates entry %q has no Chinese translation in the catalogs", tpl)
		}
	}
	for _, p := range errPhrases {
		if _, ok := messages()[p]; !ok {
			t.Errorf("errPhrases entry %q has no Chinese translation in the catalogs", p)
		}
	}
}

func TestTranslateMessage(t *testing.T) {
	t.Setenv("OLLAMA_LANG", "zh-CN")
	cases := []struct{ in, wantSub, notSub string }{
		{`unknown command "lst" for "ollama"`, "", `unknown command`},
		{`unknown command "lst" for "ollama"` + "\n\nDid you mean this?\n\tlist", "你是不是想输入", "Did you mean"},
		{`unknown flag: --bogus`, "", `unknown flag`},
		{`requires at least 1 arg(s), only received 0`, "", `requires at least`},
		{`(line 3): no FROM line`, "", `no FROM line`},
		{`model "x" not found, try pulling it first`, `model "x" not found`, ""}, // API errors pass through
	}
	for _, c := range cases {
		got := translateMessage(c.in)
		if c.wantSub != "" && !strings.Contains(got, c.wantSub) {
			t.Errorf("translateMessage(%q) = %q, want substring %q", c.in, got, c.wantSub)
		}
		if c.notSub != "" && strings.Contains(got, c.notSub) {
			t.Errorf("translateMessage(%q) = %q, must not contain %q", c.in, got, c.notSub)
		}
	}
	t.Setenv("OLLAMA_LANG", "en")
	if got := translateMessage(`unknown flag: --bogus`); got != `unknown flag: --bogus` {
		t.Errorf("English translateMessage = %q, want identity", got)
	}
}

func TestLocalizeUsageTemplateEnglishIdentity(t *testing.T) {
	t.Setenv("OLLAMA_LANG", "en")
	const tpl = "Usage:\n  ollama [command]\n\nAvailable Commands:\n  run Run a model\n\nFlags:\n  -h --help help for ollama\n"
	if got := LocalizeUsageTemplate(tpl); got != tpl {
		t.Errorf("English template changed:\ngot  %q\nwant %q", got, tpl)
	}
}

// verbClass groups printf verbs by the argument type they accept. 'v' accepts
// anything, so it is compatible with every class.
func verbClass(letter byte) string {
	switch letter {
	case 'v', 'V':
		return "any"
	case 'w':
		// %w captures already-rendered text; translations may render it as %s
		// (fillTemplate normalizes %w -> %s before formatting).
		return "string"
	case 's', 'q', 't', 'T':
		return "string"
	case 'd', 'b', 'o', 'O', 'x', 'X':
		return "int"
	case 'f', 'F', 'e', 'E', 'g', 'G':
		return "float"
	}
	return "other"
}

func compatible(a, b string) bool {
	return a == b || a == "any" || b == "any"
}

var verbRe = regexp.MustCompile(`%(?:\[[0-9]+\])?[-+#0 ']*(?:[0-9]+|\*)?(?:\.(?:[0-9]+|\*))?[a-zA-Z%]`)

// verbs extracts the verb letters of a format string, skipping %%.

func verbs(format string) []byte {
	var out []byte
	for _, m := range verbRe.FindAllString(format, -1) {
		if strings.HasSuffix(m, "%%") || m == "%%" {
			continue
		}
		out = append(out, m[len(m)-1])
	}
	return out
}

// TestFormatVerbParity enforces principle 3: every Chinese translation keeps
// the format verb sequence of its English source, so translated messages can
// never mis-render or drop arguments.
func TestFormatVerbParity(t *testing.T) {
	for _, src := range []map[string]string{
		zhHelp, zhCmd, zhChat, zhTui, zhLaunch, zhLaunch2,
		zhCreate, zhProgress, zhMisc,
	} {
		for en, zh := range src {
			enVerbs, zhVerbs := verbs(en), verbs(zh)
			if len(enVerbs) != len(zhVerbs) {
				t.Errorf("verb count mismatch:\n  en: %q → %v\n  zh: %q → %v", en, enVerbs, zh, zhVerbs)
				continue
			}
			for i := range enVerbs {
				a, b := verbClass(enVerbs[i]), verbClass(zhVerbs[i])
				if !compatible(a, b) {
					t.Errorf("verb kind mismatch at position %d:\n  en: %q (%c → %s)\n  zh: %q (%c → %s)",
						i, en, enVerbs[i], a, zh, zhVerbs[i], b)
				}
			}
		}
	}
}

// TestNoDuplicateKeys enforces principle 2: the same English string must never
// map to two different Chinese strings (and should ideally appear once).
func TestNoDuplicateKeys(t *testing.T) {
	seen := map[string]struct {
		value string
		file  string
	}{}
	for _, entry := range []struct {
		name, file string
		m          map[string]string
	}{
		{"zhHelp", "zh_help.go", zhHelp},
		{"zhCmd", "zh_cmd.go", zhCmd},
		{"zhChat", "zh_chat.go", zhChat},
		{"zhTui", "zh_tui.go", zhTui},
		{"zhLaunch", "zh_launch.go", zhLaunch},
		{"zhLaunch2", "zh_launch2.go", zhLaunch2},
		{"zhCreate", "zh_create.go", zhCreate},
		{"zhProgress", "zh_progress.go", zhProgress},
		{"zhMisc", "zh_misc.go", zhMisc},
	} {
		for en, zh := range entry.m {
			if prev, ok := seen[en]; ok {
				if prev.value != zh {
					t.Errorf("conflicting translations for %q:\n  %s: %q\n  %s: %q",
						en, prev.file, prev.value, entry.file, zh)
				} else {
					// Same English key with the same translation in two
					// catalogs is redundant but never wrong; batches may
					// legitimately both need it.
					t.Logf("duplicate key %q in %s (also in %s)", en, entry.file, prev.file)
				}
				continue
			}
			seen[en] = struct {
				value string
				file  string
			}{zh, entry.file}
		}
	}
}

// TestKeysExistInSource enforces that every catalog key really appears in the
// code base (as a quoted or raw literal), catching typoed or stale keys that
// would silently fall back to English.
func TestKeysExistInSource(t *testing.T) {
	repo := filepath.Join("..")
	sources := collectGoSources(t, repo)
	// Catalog files are excluded: otherwise a key could "match" its own entry
	// and a typo would silently pass this check.
	for en := range messages() {
		quoted := strconv.Quote(en)
		raw := "`" + en + "`"
		found := false
		for _, src := range sources {
			if strings.Contains(src, quoted) || strings.Contains(src, raw) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("catalog key not found in any source literal:\n  key: %q", en)
		}
	}
}

// TestWrappedStringsHaveTranslations enforces principle 1 (nothing that should
// be translated is left untranslated): every i18n.T("...") call site in the
// repository must have a catalog entry. Without an entry the Chinese build
// would silently print English. The scan is AST-based so it covers both
// interpreted and raw string literals.
func TestWrappedStringsHaveTranslations(t *testing.T) {
	catalog := messages()
	missing := 0
	fset := token.NewFileSet()
	filepath.Walk(filepath.Join(".."), func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		if strings.Contains(filepath.ToSlash(path), "/i18n/") {
			return nil // the package itself defines the labels it uses
		}
		f, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			return nil
		}
		ast.Inspect(f, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok || sel.Sel.Name != "T" {
				return true
			}
			pkg, ok := sel.X.(*ast.Ident)
			if !ok || pkg.Name != "i18n" || len(call.Args) != 1 {
				return true
			}
			lit, ok := call.Args[0].(*ast.BasicLit)
			if !ok || lit.Kind != token.STRING {
				// Dynamic keys (e.g. i18n.T(e.Description)) cannot be checked
				// statically; coverage for them is asserted by dedicated tests.
				return true
			}
			key, err := strconv.Unquote(lit.Value)
			if err != nil {
				return true
			}
			if _, ok := catalog[key]; !ok {
				missing++
				t.Errorf("%s: i18n.T(%q) has no catalog entry", path, key)
			}
			return true
		})
		return nil
	})
	if missing == 0 && len(catalog) == 0 {
		t.Log("no translations yet")
	}
}

// TestEnvDescriptionsTranslated asserts every environment-variable description
// rendered by `appendEnvDocs` has a catalog entry (the T() call there takes a
// variable, so the literal scan cannot verify it).
func TestEnvDescriptionsTranslated(t *testing.T) {
	for key, e := range envconfig.AsMap() {
		if e.Description == "" {
			continue
		}
		if _, ok := messages()[e.Description]; !ok {
			t.Errorf("env description for %s has no catalog entry:\n  %q", key, e.Description)
		}
	}
}

// TestShowSectionHeadersTranslated guards the dynamic i18n.T(header) display
// site in showInfo: every section header passed to tableRender must have a
// catalog entry (the literal is not wrapped, so the call-site scan misses it).
func TestShowSectionHeadersTranslated(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "cmd", "cmd.go"))
	if err != nil {
		t.Fatal(err)
	}
	headers := regexp.MustCompile(`tableRender\(\s*"([^"]+)"`).FindAllStringSubmatch(string(data), -1)
	if len(headers) == 0 {
		t.Fatal("no tableRender call sites found")
	}
	for _, m := range headers {
		if _, ok := messages()[m[1]]; !ok {
			t.Errorf("showInfo section header %q has no catalog entry", m[1])
		}
	}
}

// TestHelpForUsage covers the generated --help flag usage line.
func TestHelpForUsage(t *testing.T) {
	t.Setenv("OLLAMA_LANG", "en")
	if got := HelpForUsage("run"); got != "help for run" {
		t.Errorf("English HelpForUsage = %q", got)
	}
	t.Setenv("OLLAMA_LANG", "zh-CN")
	if got := HelpForUsage("run"); got == "help for run" {
		t.Errorf("Chinese HelpForUsage not translated: %q", got)
	}
	if got := HelpForUsage(""); got == "" {
		t.Errorf("empty name must fall back to this command")
	}
}

func TestMatchTemplate(t *testing.T) {
	captures, rest, ok := matchTemplate(`unknown command %q for %q`, `unknown command "lst" for "ollama"`+"\n\nDid you mean this?\n\tlist")
	if !ok {
		t.Fatalf("template did not match")
	}
	if len(captures) != 2 || captures[0] != "lst" || captures[1] != "ollama" {
		t.Errorf("captures = %#v", captures)
	}
	if !strings.HasPrefix(rest, "\n\n") {
		t.Errorf("rest = %q, want the suggestion block", rest)
	}

	captures, _, ok = matchTemplate(`requires at least %d arg(s), only received %d`, `requires at least 1 arg(s), only received 0`)
	if !ok || len(captures) != 2 || captures[0] != "1" || captures[1] != "0" {
		t.Errorf("digit captures = %#v ok=%v", captures, ok)
	}

	if _, _, ok := matchTemplate(`unknown flag: --%s`, `some other message`); ok {
		t.Errorf("must not match unrelated messages")
	}
}

func TestFillTemplate(t *testing.T) {
	got := fillTemplate(`未知命令 %q（%q）`, []string{"lst", "ollama"})
	if !strings.Contains(got, "lst") || !strings.Contains(got, "ollama") {
		t.Errorf("fillTemplate = %q", got)
	}
	// %w captures arrive as rendered text; fillTemplate must not emit %!w(...)
	got = fillTemplate(`无法找到用户 “%s”：%s`, []string{"alice", "no such user"})
	if !strings.Contains(got, "alice") || !strings.Contains(got, "no such user") || strings.Contains(got, "%!") {
		t.Errorf("fillTemplate %%w handling = %q", got)
	}
	// a translation that keeps %w is also normalized to %s
	got = fillTemplate(`failed: %w`, []string{"boom"})
	if got != "failed: boom" {
		t.Errorf("fillTemplate %%w normalize = %q", got)
	}
	got = fillTemplate(`至少需要 %d 个参数`, []string{"1"})
	if got != "至少需要 1 个参数" {
		t.Errorf("fillTemplate int = %q", got)
	}
}

// collectGoSources returns the contents of every non-test .go file under root.
func collectGoSources(t *testing.T, root string) []string {
	t.Helper()
	fset := token.NewFileSet()
	var out []string
	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !strings.HasSuffix(path, ".go") {
			return nil
		}
		// Skip the catalog files themselves: a key must exist in real source,
		// not merely in its own translation entry.
		if strings.Contains(filepath.ToSlash(path), "/i18n/zh_") {
			return nil
		}
		if _, err := parser.ParseFile(fset, path, nil, parser.ParseComments); err != nil {
			return nil // skip files that do not parse (generated/vendor)
		}
		data, err := os.ReadFile(path)
		if err == nil {
			out = append(out, string(data))
		}
		return nil
	})
	if len(out) == 0 {
		t.Fatalf("no Go sources found under %s", root)
	}
	return out
}
