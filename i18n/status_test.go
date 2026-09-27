package i18n

import "testing"

func TestStatusIdempotent(t *testing.T) {
	t.Setenv("OLLAMA_LANG", "zh-CN")
	// already-translated input must pass through unchanged
	if got := Status("拉取 manifest"); got != "拉取 manifest" {
		t.Errorf("double translation: %q", got)
	}
	if got := Status("复制文件 abc 0%"); got != "复制文件 abc 0%" {
		t.Errorf("double translation: %q", got)
	}
	// English static + dynamic still translate
	if got := Status("gathering model components"); got == "gathering model components" {
		t.Errorf("static status untranslated")
	}
	if got := Status("pulling abc123..."); got != "拉取 abc123..." {
		t.Errorf("dynamic status = %q", got)
	}
}
