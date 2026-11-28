package parsers

import "testing"

// paolo-perrone's review case on #19904: the model opens a second call
// before closing the first. The boundary is the first opener outside every
// <arg_value> span, so read() must keep its own argument and not absorb
// the second call's pairs.
func TestLagunaSecondOpenerBeforeCloseIsCallBoundary(t *testing.T) {
	p := &LagunaParser{}
	p.Init(readWriteToolsLaguna(), nil, nil)

	input := "<tool_call>read\n<arg_key>path</arg_key>\n<arg_value>notes.txt</arg_value>\n<tool_call>write\n<arg_key>path</arg_key>\n<arg_value>out.txt</arg_value>\n<arg_key>content</arg_key>\n<arg_value>hi</arg_value>\n</tool_call>"
	_, _, calls, err := p.Add(input, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	if calls[0].Function.Name != "read" {
		t.Fatalf("name = %q, want read", calls[0].Function.Name)
	}
	path, ok := calls[0].Function.Arguments.Get("path")
	if !ok || path != "notes.txt" {
		t.Errorf("path = %v/%v, want notes.txt", path, ok)
	}
	if _, ok := calls[0].Function.Arguments.Get("content"); ok {
		t.Error("read() must not gain a content key from the second call")
	}
}

// A nested opener *inside* an arg_value still belongs to the value.
func TestLagunaNestedOpenerInsideValueStillKept(t *testing.T) {
	p := &LagunaParser{}
	p.Init(sayToolLaguna(), nil, nil)

	input := "<tool_call>say\n<arg_key>text</arg_key>\n<arg_value>before <tool_call> after</arg_value>\n</tool_call>"
	_, _, calls, err := p.Add(input, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	got, _ := calls[0].Function.Arguments.Get("text")
	if got != "before <tool_call> after" {
		t.Errorf("text = %q, want %q", got, "before <tool_call> after")
	}
}
