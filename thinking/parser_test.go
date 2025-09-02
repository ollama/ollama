package thinking

import (
	"testing"
)

func TestExtractThinking(t *testing.T) {
	tests := []struct {
		in, wantContent, wantThink string
	}{
		{
			in:          "<think> internal </think> world",
			wantThink:   "internal ",
			wantContent: "world",
		},
		{
			in:          "<think>a</think><think>b</think>c",
			wantThink:   "a",
			wantContent: "<think>b</think>c",
		},
		{
			in:          "no think",
			wantThink:   "",
			wantContent: "no think",
		},
	}
	for i, tt := range tests {
		parser := Parser{
			OpeningTag: "<think>",
			ClosingTag: "</think>",
		}
		gotThinking, gotContent := parser.AddContent(tt.in)
		if gotContent != tt.wantContent || gotThinking != tt.wantThink {
			t.Errorf("case %d: got (%q,%q), want (%q,%q)", i, gotThinking, gotContent, tt.wantThink, tt.wantContent)
		}
	}
}

func TestThinkingStreaming(t *testing.T) {
	type step struct {
		input          string
		wantThinking   string
		wantContent    string
		wantStateAfter thinkingState
	}

	cases := []struct {
		desc  string
		skip  bool
		steps []step
	}{
		{
			desc: "content without a thinking tag",
			steps: []step{
				{
					input:          "  abc",
					wantThinking:   "",
					wantContent:    "  abc",
					wantStateAfter: thinkingState_ThinkingDone,
				},
				// regression test for a bug where we were transitioning directly to
				// ThinkingDone without clearing the buffer. This would cuase the first
				// step to be outputted twice
				{
					input:          "def",
					wantThinking:   "",
					wantContent:    "def",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
		{
			desc: "content before a thinking tag nerfs the thinking tag",
			steps: []step{
				{
					input:          "  abc <think>def</think> ghi",
					wantThinking:   "",
					wantContent:    "  abc <think>def</think> ghi",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
		{
			desc: "building up a thinking tag partially",
			steps: []step{
				{
					input:          "  <th",
					wantThinking:   "",
					wantContent:    "",
					wantStateAfter: thinkingState_LookingForOpening,
				},
				{
					input:          "in",
					wantThinking:   "",
					wantContent:    "",
					wantStateAfter: thinkingState_LookingForOpening,
				},
				{
					input:          "k>a",
					wantThinking:   "a",
					wantContent:    "",
					wantStateAfter: thinkingState_Thinking,
				},
			},
		},
		{
			desc: "partial closing tag",
			steps: []step{
				{
					input:          "<think>abc</th",
					wantThinking:   "abc",
					wantContent:    "",
					wantStateAfter: thinkingState_Thinking,
				},
				{
					input:          "ink>def",
					wantThinking:   "",
					wantContent:    "def",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
		{
			desc: "partial closing tag fakeout",
			steps: []step{
				{
					input:          "<think>abc</th",
					wantThinking:   "abc",
					wantContent:    "",
					wantStateAfter: thinkingState_Thinking,
				},
				{
					input:          "ing>def",
					wantThinking:   "</thing>def",
					wantContent:    "",
					wantStateAfter: thinkingState_Thinking,
				},
				{
					input:          "ghi</thi",
					wantThinking:   "ghi",
					wantContent:    "",
					wantStateAfter: thinkingState_Thinking,
				},
				{
					input:          "nk>jkl",
					wantThinking:   "",
					wantContent:    "jkl",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
		{
			desc: "whitespace after thinking tag",
			steps: []step{
				{
					input:          "  <think>abc</think>\n\ndef",
					wantThinking:   "abc",
					wantContent:    "def",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
		{
			desc: "whitespace after thinking tag (incremental)",
			steps: []step{
				{
					input:          "  <think>abc</think>",
					wantThinking:   "abc",
					wantContent:    "",
					wantStateAfter: thinkingState_ThinkingDoneEatingWhitespace,
				},
				{
					input:          "\n\ndef",
					wantThinking:   "",
					wantContent:    "def",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
		{
			desc: "whitespace after thinking tag with content and more whitespace",
			steps: []step{
				{
					input:          "  <think>abc</think>\n\ndef ",
					wantThinking:   "abc",
					wantContent:    "def ",
					wantStateAfter: thinkingState_ThinkingDone,
				},
				{
					input:          " ghi",
					wantThinking:   "",
					wantContent:    " ghi",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
		{
			desc: "token by token",
			steps: []step{
				{
					input:          "<think>",
					wantThinking:   "",
					wantContent:    "",
					wantStateAfter: thinkingState_ThinkingStartedEatingWhitespace,
				},
				{
					input:          "\n",
					wantThinking:   "",
					wantContent:    "",
					wantStateAfter: thinkingState_ThinkingStartedEatingWhitespace,
				},
				{
					input:          "</think>",
					wantThinking:   "",
					wantContent:    "",
					wantStateAfter: thinkingState_ThinkingDoneEatingWhitespace,
				},
				{
					input:          "\n\n",
					wantThinking:   "",
					wantContent:    "",
					wantStateAfter: thinkingState_ThinkingDoneEatingWhitespace,
				},
				{
					input:          "Hi",
					wantThinking:   "",
					wantContent:    "Hi",
					wantStateAfter: thinkingState_ThinkingDone,
				},
				{
					input:          " there",
					wantThinking:   "",
					wantContent:    " there",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
		{
			desc: "leading thinking whitespace",
			steps: []step{
				{
					input:          "  <think>   \t ",
					wantThinking:   "",
					wantContent:    "",
					wantStateAfter: thinkingState_ThinkingStartedEatingWhitespace,
				},
				{
					input:          "  these are some ",
					wantThinking:   "these are some ",
					wantContent:    "",
					wantStateAfter: thinkingState_Thinking,
				},
				{
					input:          "thoughts </think>  ",
					wantThinking:   "thoughts ",
					wantContent:    "",
					wantStateAfter: thinkingState_ThinkingDoneEatingWhitespace,
				},
				{
					input:          "  more content",
					wantThinking:   "",
					wantContent:    "more content",
					wantStateAfter: thinkingState_ThinkingDone,
				},
			},
		},
	}

	for _, c := range cases {
		parser := Parser{
			OpeningTag: "<think>",
			ClosingTag: "</think>",
		}
		if c.skip {
			continue
		}
		for i, step := range c.steps {
			thinking, content := parser.AddContent(step.input)
			if content != step.wantContent || thinking != step.wantThinking {
				t.Errorf("case %q (step %d): got (%q,%q), want (%q,%q)", c.desc, i, content, thinking, step.wantContent, step.wantThinking)
			}
			if parser.state != step.wantStateAfter {
				t.Errorf("case %q (step %d): got state %s, want %s", c.desc, i, parser.state, step.wantStateAfter)
			}
		}
	}
}
