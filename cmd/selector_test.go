package cmd

import (
	"bytes"
	"strings"
	"testing"
)

func TestFilterItems(t *testing.T) {
	items := []SelectItem{
		{Name: "llama3.2:latest"},
		{Name: "qwen2.5:7b"},
		{Name: "deepseek-v3:cloud"},
		{Name: "GPT-OSS:20b"},
	}

	t.Run("EmptyFilter_ReturnsAllItems", func(t *testing.T) {
		result := filterItems(items, "")
		if len(result) != len(items) {
			t.Errorf("expected %d items, got %d", len(items), len(result))
		}
	})

	t.Run("CaseInsensitive_UppercaseFilterMatchesLowercase", func(t *testing.T) {
		result := filterItems(items, "LLAMA")
		if len(result) != 1 || result[0].Name != "llama3.2:latest" {
			t.Errorf("expected llama3.2:latest, got %v", result)
		}
	})

	t.Run("CaseInsensitive_LowercaseFilterMatchesUppercase", func(t *testing.T) {
		result := filterItems(items, "gpt")
		if len(result) != 1 || result[0].Name != "GPT-OSS:20b" {
			t.Errorf("expected GPT-OSS:20b, got %v", result)
		}
	})

	t.Run("PartialMatch", func(t *testing.T) {
		result := filterItems(items, "deep")
		if len(result) != 1 || result[0].Name != "deepseek-v3:cloud" {
			t.Errorf("expected deepseek-v3:cloud, got %v", result)
		}
	})

	t.Run("NoMatch_ReturnsEmpty", func(t *testing.T) {
		result := filterItems(items, "nonexistent")
		if len(result) != 0 {
			t.Errorf("expected 0 items, got %d", len(result))
		}
	})
}

func TestSelectState(t *testing.T) {
	items := []SelectItem{
		{Name: "item1"},
		{Name: "item2"},
		{Name: "item3"},
	}

	t.Run("InitialState", func(t *testing.T) {
		s := newSelectState(items)
		if s.selected != 0 {
			t.Errorf("expected selected=0, got %d", s.selected)
		}
		if s.filter != "" {
			t.Errorf("expected empty filter, got %q", s.filter)
		}
		if s.scrollOffset != 0 {
			t.Errorf("expected scrollOffset=0, got %d", s.scrollOffset)
		}
	})

	t.Run("Enter_SelectsCurrentItem", func(t *testing.T) {
		s := newSelectState(items)
		done, result, err := s.handleInput(eventEnter, 0)
		if !done || result != "item1" || err != nil {
			t.Errorf("expected (true, item1, nil), got (%v, %v, %v)", done, result, err)
		}
	})

	t.Run("Enter_WithFilter_SelectsFilteredItem", func(t *testing.T) {
		s := newSelectState(items)
		s.filter = "item3"
		done, result, err := s.handleInput(eventEnter, 0)
		if !done || result != "item3" || err != nil {
			t.Errorf("expected (true, item3, nil), got (%v, %v, %v)", done, result, err)
		}
	})

	t.Run("Enter_EmptyFilteredList_DoesNothing", func(t *testing.T) {
		s := newSelectState(items)
		s.filter = "nonexistent"
		done, result, err := s.handleInput(eventEnter, 0)
		if done || result != "" || err != nil {
			t.Errorf("expected (false, '', nil), got (%v, %v, %v)", done, result, err)
		}
	})

	t.Run("Escape_ReturnsCancelledError", func(t *testing.T) {
		s := newSelectState(items)
		done, result, err := s.handleInput(eventEscape, 0)
		if !done || result != "" || err != ErrCancelled {
			t.Errorf("expected (true, '', ErrCancelled), got (%v, %v, %v)", done, result, err)
		}
	})

	t.Run("Down_MovesSelection", func(t *testing.T) {
		s := newSelectState(items)
		s.handleInput(eventDown, 0)
		if s.selected != 1 {
			t.Errorf("expected selected=1, got %d", s.selected)
		}
	})

	t.Run("Down_AtBottom_StaysAtBottom", func(t *testing.T) {
		s := newSelectState(items)
		s.selected = 2
		s.handleInput(eventDown, 0)
		if s.selected != 2 {
			t.Errorf("expected selected=2 (stayed at bottom), got %d", s.selected)
		}
	})

	t.Run("Up_MovesSelection", func(t *testing.T) {
		s := newSelectState(items)
		s.selected = 2
		s.handleInput(eventUp, 0)
		if s.selected != 1 {
			t.Errorf("expected selected=1, got %d", s.selected)
		}
	})

	t.Run("Up_AtTop_StaysAtTop", func(t *testing.T) {
		s := newSelectState(items)
		s.handleInput(eventUp, 0)
		if s.selected != 0 {
			t.Errorf("expected selected=0 (stayed at top), got %d", s.selected)
		}
	})

	t.Run("Char_AppendsToFilter", func(t *testing.T) {
		s := newSelectState(items)
		s.handleInput(eventChar, 'i')
		s.handleInput(eventChar, 't')
		s.handleInput(eventChar, 'e')
		s.handleInput(eventChar, 'm')
		s.handleInput(eventChar, '2')
		if s.filter != "item2" {
			t.Errorf("expected filter='item2', got %q", s.filter)
		}
		filtered := s.filtered()
		if len(filtered) != 1 || filtered[0].Name != "item2" {
			t.Errorf("expected [item2], got %v", filtered)
		}
	})

	t.Run("Char_ResetsSelectionToZero", func(t *testing.T) {
		s := newSelectState(items)
		s.selected = 2
		s.handleInput(eventChar, 'x')
		if s.selected != 0 {
			t.Errorf("expected selected=0 after typing, got %d", s.selected)
		}
	})

	t.Run("Backspace_RemovesLastFilterChar", func(t *testing.T) {
		s := newSelectState(items)
		s.filter = "test"
		s.handleInput(eventBackspace, 0)
		if s.filter != "tes" {
			t.Errorf("expected filter='tes', got %q", s.filter)
		}
	})

	t.Run("Backspace_EmptyFilter_DoesNothing", func(t *testing.T) {
		s := newSelectState(items)
		s.handleInput(eventBackspace, 0)
		if s.filter != "" {
			t.Errorf("expected filter='', got %q", s.filter)
		}
	})

	t.Run("Backspace_ResetsSelectionToZero", func(t *testing.T) {
		s := newSelectState(items)
		s.filter = "test"
		s.selected = 2
		s.handleInput(eventBackspace, 0)
		if s.selected != 0 {
			t.Errorf("expected selected=0 after backspace, got %d", s.selected)
		}
	})

	t.Run("Scroll_DownPastVisibleItems_ScrollsViewport", func(t *testing.T) {
		// maxDisplayedItems is 10, so with 15 items we need to scroll
		manyItems := make([]SelectItem, 15)
		for i := range manyItems {
			manyItems[i] = SelectItem{Name: string(rune('a' + i))}
		}
		s := newSelectState(manyItems)

		// move down 12 times (past the 10-item viewport)
		for i := 0; i < 12; i++ {
			s.handleInput(eventDown, 0)
		}

		if s.selected != 12 {
			t.Errorf("expected selected=12, got %d", s.selected)
		}
		if s.scrollOffset != 3 {
			t.Errorf("expected scrollOffset=3 (12-10+1), got %d", s.scrollOffset)
		}
	})

	t.Run("Scroll_UpPastScrollOffset_ScrollsViewport", func(t *testing.T) {
		manyItems := make([]SelectItem, 15)
		for i := range manyItems {
			manyItems[i] = SelectItem{Name: string(rune('a' + i))}
		}
		s := newSelectState(manyItems)
		s.selected = 5
		s.scrollOffset = 5

		s.handleInput(eventUp, 0)

		if s.selected != 4 {
			t.Errorf("expected selected=4, got %d", s.selected)
		}
		if s.scrollOffset != 4 {
			t.Errorf("expected scrollOffset=4, got %d", s.scrollOffset)
		}
	})
}

func TestMultiSelectState(t *testing.T) {
	items := []SelectItem{
		{Name: "item1"},
		{Name: "item2"},
		{Name: "item3"},
	}

	t.Run("InitialState_NoPrechecked", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		if s.highlighted != 0 {
			t.Errorf("expected highlighted=0, got %d", s.highlighted)
		}
		if s.selectedCount() != 0 {
			t.Errorf("expected 0 selected, got %d", s.selectedCount())
		}
		if s.focusOnButton {
			t.Error("expected focusOnButton=false initially")
		}
	})

	t.Run("InitialState_WithPrechecked", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item2", "item3"})
		if s.selectedCount() != 2 {
			t.Errorf("expected 2 selected, got %d", s.selectedCount())
		}
		if !s.isChecked(1) || !s.isChecked(2) {
			t.Error("expected item2 and item3 to be checked")
		}
	})

	t.Run("Prechecked_PreservesSelectionOrder", func(t *testing.T) {
		// order matters: first checked = default model
		s := newMultiSelectState(items, []string{"item3", "item1"})
		if len(s.checkOrder) != 2 {
			t.Fatalf("expected 2 in checkOrder, got %d", len(s.checkOrder))
		}
		if s.checkOrder[0] != 2 || s.checkOrder[1] != 0 {
			t.Errorf("expected checkOrder=[2,0] (item3 first), got %v", s.checkOrder)
		}
	})

	t.Run("Prechecked_IgnoresInvalidNames", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1", "nonexistent"})
		if s.selectedCount() != 1 {
			t.Errorf("expected 1 selected (nonexistent ignored), got %d", s.selectedCount())
		}
	})

	t.Run("Toggle_ChecksUncheckedItem", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		s.toggleItem()
		if !s.isChecked(0) {
			t.Error("expected item1 to be checked after toggle")
		}
	})

	t.Run("Toggle_UnchecksCheckedItem", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1"})
		s.toggleItem()
		if s.isChecked(0) {
			t.Error("expected item1 to be unchecked after toggle")
		}
	})

	t.Run("Toggle_RemovesFromCheckOrder", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1", "item2", "item3"})
		s.highlighted = 1 // toggle item2
		s.toggleItem()

		if len(s.checkOrder) != 2 {
			t.Fatalf("expected 2 in checkOrder, got %d", len(s.checkOrder))
		}
		// should be [0, 2] (item1, item3) with item2 removed
		if s.checkOrder[0] != 0 || s.checkOrder[1] != 2 {
			t.Errorf("expected checkOrder=[0,2], got %v", s.checkOrder)
		}
	})

	t.Run("Enter_TogglesWhenNotOnButton", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		s.handleInput(eventEnter, 0)
		if !s.isChecked(0) {
			t.Error("expected item1 to be checked after enter")
		}
	})

	t.Run("Enter_OnButton_ReturnsSelection", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item2", "item1"})
		s.focusOnButton = true

		done, result, err := s.handleInput(eventEnter, 0)

		if !done || err != nil {
			t.Errorf("expected done=true, err=nil, got done=%v, err=%v", done, err)
		}
		// result should preserve selection order
		if len(result) != 2 || result[0] != "item2" || result[1] != "item1" {
			t.Errorf("expected [item2, item1], got %v", result)
		}
	})

	t.Run("Enter_OnButton_EmptySelection_DoesNothing", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		s.focusOnButton = true
		done, result, err := s.handleInput(eventEnter, 0)
		if done || result != nil || err != nil {
			t.Errorf("expected (false, nil, nil), got (%v, %v, %v)", done, result, err)
		}
	})

	t.Run("Tab_SwitchesToButton_WhenHasSelection", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1"})
		s.handleInput(eventTab, 0)
		if !s.focusOnButton {
			t.Error("expected focus on button after tab")
		}
	})

	t.Run("Tab_DoesNothing_WhenNoSelection", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		s.handleInput(eventTab, 0)
		if s.focusOnButton {
			t.Error("tab should not focus button when nothing selected")
		}
	})

	t.Run("Tab_TogglesButtonFocus", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1"})
		s.handleInput(eventTab, 0)
		if !s.focusOnButton {
			t.Error("expected focus on button after first tab")
		}
		s.handleInput(eventTab, 0)
		if s.focusOnButton {
			t.Error("expected focus back on list after second tab")
		}
	})

	t.Run("Escape_ReturnsCancelledError", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1"})
		done, result, err := s.handleInput(eventEscape, 0)
		if !done || result != nil || err != ErrCancelled {
			t.Errorf("expected (true, nil, ErrCancelled), got (%v, %v, %v)", done, result, err)
		}
	})

	t.Run("IsDefault_TrueForFirstChecked", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item2", "item1"})
		if !s.isDefault(1) {
			t.Error("expected item2 (idx 1) to be default (first checked)")
		}
		if s.isDefault(0) {
			t.Error("expected item1 (idx 0) to NOT be default")
		}
	})

	t.Run("IsDefault_FalseWhenNothingChecked", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		if s.isDefault(0) {
			t.Error("expected isDefault=false when nothing checked")
		}
	})

	t.Run("Down_MovesHighlight", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		s.handleInput(eventDown, 0)
		if s.highlighted != 1 {
			t.Errorf("expected highlighted=1, got %d", s.highlighted)
		}
	})

	t.Run("Up_MovesHighlight", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		s.highlighted = 1
		s.handleInput(eventUp, 0)
		if s.highlighted != 0 {
			t.Errorf("expected highlighted=0, got %d", s.highlighted)
		}
	})

	t.Run("Arrow_ReturnsFocusFromButton", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1"})
		s.focusOnButton = true
		s.handleInput(eventDown, 0)
		if s.focusOnButton {
			t.Error("expected focus to return to list on arrow key")
		}
	})

	t.Run("Char_AppendsToFilter", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		s.handleInput(eventChar, 'x')
		if s.filter != "x" {
			t.Errorf("expected filter='x', got %q", s.filter)
		}
	})

	t.Run("Char_ResetsHighlightAndScroll", func(t *testing.T) {
		manyItems := make([]SelectItem, 15)
		for i := range manyItems {
			manyItems[i] = SelectItem{Name: string(rune('a' + i))}
		}
		s := newMultiSelectState(manyItems, nil)
		s.highlighted = 10
		s.scrollOffset = 5

		s.handleInput(eventChar, 'x')

		if s.highlighted != 0 {
			t.Errorf("expected highlighted=0, got %d", s.highlighted)
		}
		if s.scrollOffset != 0 {
			t.Errorf("expected scrollOffset=0, got %d", s.scrollOffset)
		}
	})

	t.Run("Backspace_RemovesLastFilterChar", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		s.filter = "test"
		s.handleInput(eventBackspace, 0)
		if s.filter != "tes" {
			t.Errorf("expected filter='tes', got %q", s.filter)
		}
	})

	t.Run("Backspace_RemovesFocusFromButton", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1"})
		s.filter = "x"
		s.focusOnButton = true
		s.handleInput(eventBackspace, 0)
		if s.focusOnButton {
			t.Error("expected focusOnButton=false after backspace")
		}
	})
}

func TestParseInput(t *testing.T) {
	t.Run("Enter", func(t *testing.T) {
		event, char, err := parseInput(bytes.NewReader([]byte{13}))
		if err != nil || event != eventEnter || char != 0 {
			t.Errorf("expected (eventEnter, 0, nil), got (%v, %v, %v)", event, char, err)
		}
	})

	t.Run("Escape", func(t *testing.T) {
		event, _, err := parseInput(bytes.NewReader([]byte{27}))
		if err != nil || event != eventEscape {
			t.Errorf("expected eventEscape, got %v", event)
		}
	})

	t.Run("CtrlC_TreatedAsEscape", func(t *testing.T) {
		event, _, err := parseInput(bytes.NewReader([]byte{3}))
		if err != nil || event != eventEscape {
			t.Errorf("expected eventEscape for Ctrl+C, got %v", event)
		}
	})

	t.Run("Tab", func(t *testing.T) {
		event, _, err := parseInput(bytes.NewReader([]byte{9}))
		if err != nil || event != eventTab {
			t.Errorf("expected eventTab, got %v", event)
		}
	})

	t.Run("Backspace", func(t *testing.T) {
		event, _, err := parseInput(bytes.NewReader([]byte{127}))
		if err != nil || event != eventBackspace {
			t.Errorf("expected eventBackspace, got %v", event)
		}
	})

	t.Run("UpArrow", func(t *testing.T) {
		event, _, err := parseInput(bytes.NewReader([]byte{27, 91, 65}))
		if err != nil || event != eventUp {
			t.Errorf("expected eventUp, got %v", event)
		}
	})

	t.Run("DownArrow", func(t *testing.T) {
		event, _, err := parseInput(bytes.NewReader([]byte{27, 91, 66}))
		if err != nil || event != eventDown {
			t.Errorf("expected eventDown, got %v", event)
		}
	})

	t.Run("PrintableChars", func(t *testing.T) {
		tests := []struct {
			name string
			char byte
		}{
			{"lowercase", 'a'},
			{"uppercase", 'Z'},
			{"digit", '5'},
			{"space", ' '},
			{"tilde", '~'},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				event, char, err := parseInput(bytes.NewReader([]byte{tt.char}))
				if err != nil || event != eventChar || char != tt.char {
					t.Errorf("expected (eventChar, %q), got (%v, %q)", tt.char, event, char)
				}
			})
		}
	})
}

func TestRenderSelect(t *testing.T) {
	items := []SelectItem{
		{Name: "item1", Description: "first item"},
		{Name: "item2"},
	}

	t.Run("ShowsPromptAndItems", func(t *testing.T) {
		s := newSelectState(items)
		var buf bytes.Buffer
		lineCount := renderSelect(&buf, "Select:", s)

		output := buf.String()
		if !strings.Contains(output, "Select:") {
			t.Error("expected prompt in output")
		}
		if !strings.Contains(output, "item1") {
			t.Error("expected item1 in output")
		}
		if !strings.Contains(output, "first item") {
			t.Error("expected description in output")
		}
		if !strings.Contains(output, "item2") {
			t.Error("expected item2 in output")
		}
		if lineCount != 3 { // 1 prompt + 2 items
			t.Errorf("expected 3 lines, got %d", lineCount)
		}
	})

	t.Run("EmptyFilteredList_ShowsNoMatches", func(t *testing.T) {
		s := newSelectState(items)
		s.filter = "xyz"
		var buf bytes.Buffer
		renderSelect(&buf, "Select:", s)

		if !strings.Contains(buf.String(), "no matches") {
			t.Error("expected 'no matches' message")
		}
	})

	t.Run("LongList_ShowsRemainingCount", func(t *testing.T) {
		manyItems := make([]SelectItem, 15)
		for i := range manyItems {
			manyItems[i] = SelectItem{Name: string(rune('a' + i))}
		}
		s := newSelectState(manyItems)
		var buf bytes.Buffer
		renderSelect(&buf, "Select:", s)

		// 15 items - 10 displayed = 5 more
		if !strings.Contains(buf.String(), "5 more") {
			t.Error("expected '5 more' indicator")
		}
	})
}

func TestRenderMultiSelect(t *testing.T) {
	items := []SelectItem{
		{Name: "item1"},
		{Name: "item2"},
	}

	t.Run("ShowsCheckboxes", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1"})
		var buf bytes.Buffer
		renderMultiSelect(&buf, "Select:", s)

		output := buf.String()
		if !strings.Contains(output, "[x]") {
			t.Error("expected checked checkbox [x]")
		}
		if !strings.Contains(output, "[ ]") {
			t.Error("expected unchecked checkbox [ ]")
		}
	})

	t.Run("ShowsDefaultMarker", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1"})
		var buf bytes.Buffer
		renderMultiSelect(&buf, "Select:", s)

		if !strings.Contains(buf.String(), "(default)") {
			t.Error("expected (default) marker for first checked item")
		}
	})

	t.Run("ShowsSelectedCount", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1", "item2"})
		var buf bytes.Buffer
		renderMultiSelect(&buf, "Select:", s)

		if !strings.Contains(buf.String(), "2 selected") {
			t.Error("expected '2 selected' in output")
		}
	})

	t.Run("NoSelection_ShowsHelperText", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		var buf bytes.Buffer
		renderMultiSelect(&buf, "Select:", s)

		if !strings.Contains(buf.String(), "Select at least one") {
			t.Error("expected 'Select at least one' helper text")
		}
	})
}

func TestErrCancelled(t *testing.T) {
	t.Run("NotNil", func(t *testing.T) {
		if ErrCancelled == nil {
			t.Error("ErrCancelled should not be nil")
		}
	})

	t.Run("Message", func(t *testing.T) {
		if ErrCancelled.Error() != "cancelled" {
			t.Errorf("expected 'cancelled', got %q", ErrCancelled.Error())
		}
	})
}
