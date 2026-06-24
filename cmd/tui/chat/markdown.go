package chat

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
	"sync"
	"unicode"

	"github.com/charmbracelet/glamour"
	glamouransi "github.com/charmbracelet/glamour/ansi"
	"github.com/charmbracelet/glamour/styles"
	"github.com/charmbracelet/lipgloss"
)

var markdownLinkPattern = regexp.MustCompile(`\[([^\]]+)\]\((https?://[^)\s]+)\)`)

var (
	markdownTableSeparatorPattern = regexp.MustCompile(`^:?-{3,}:?$`)
	markdownHeadingPattern        = regexp.MustCompile(`^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$`)
	markdownHorizontalRulePattern = regexp.MustCompile(`^\s{0,3}(-{3,}|\*{3,}|_{3,})\s*$`)
)

const maxMarkdownRendererCacheEntries = 8

var chatMarkdownRenderers = newMarkdownRendererCache()

type markdownRendererCache struct {
	mu        sync.Mutex
	renderers map[int]*cachedMarkdownRenderer
	order     []int
}

func newMarkdownRendererCache() *markdownRendererCache {
	return &markdownRendererCache{renderers: make(map[int]*cachedMarkdownRenderer)}
}

type cachedMarkdownRenderer struct {
	renderer *glamour.TermRenderer
	mu       sync.Mutex
}

func renderMarkdownForView(markdown string, width int) string {
	if strings.TrimSpace(markdown) == "" {
		return markdown
	}
	if width < 20 {
		width = 20
	}
	return renderMarkdownBlocks(markdownBlocks(markdown), width)
}

func exposeMarkdownLinks(markdown string) string {
	return markdownLinkPattern.ReplaceAllString(markdown, "$1 ($2)")
}

func renderMarkdownChunk(markdown string, width int) string {
	renderer, err := markdownRendererForWidth(width)
	if err != nil {
		return strings.Join(wrapChatText(markdown, width), "\n")
	}
	renderer.mu.Lock()
	rendered, err := renderer.renderer.Render(markdown)
	renderer.mu.Unlock()
	if err != nil {
		return strings.Join(wrapChatText(markdown, width), "\n")
	}
	return trimRenderedLines(rendered)
}

type markdownBlockKind int

type markdownBlock struct {
	kind  markdownBlockKind
	text  string
	table markdownTable
}

type markdownTable struct {
	headers []string
	rows    [][]string
}

func markdownBlocks(markdown string) []markdownBlock {
	lines := strings.Split(markdown, "\n")
	blocks := make([]markdownBlock, 0, 1)
	var prose []string

	flushProse := func() {
		if len(prose) == 0 {
			return
		}
		blocks = append(blocks, markdownBlock{kind: markdownBlockProse, text: strings.Join(prose, "\n")})
		prose = nil
	}

	for i := 0; i < len(lines); {
		if fence, info, ok := markdownFenceStart(lines[i]); ok {
			flushProse()
			if markdownFenceIsDiff(info) {
				var diffLines []string
				i++
				for ; i < len(lines); i++ {
					if fenceEnd(lines[i], fence) {
						break
					}
					diffLines = append(diffLines, lines[i])
				}
				if i < len(lines) {
					i++
				}
				blocks = append(blocks, markdownBlock{kind: markdownBlockDiffFence, text: strings.Join(diffLines, "\n")})
				continue
			}

			codeLines := []string{lines[i]}
			i++
			for ; i < len(lines); i++ {
				codeLines = append(codeLines, lines[i])
				if fenceEnd(lines[i], fence) {
					i++
					break
				}
			}
			blocks = append(blocks, markdownBlock{kind: markdownBlockCodeFence, text: strings.Join(codeLines, "\n")})
			continue
		}

		if table, next, ok := markdownTableAt(lines, i); ok {
			flushProse()
			blocks = append(blocks, markdownBlock{kind: markdownBlockTable, table: table})
			i = next
			continue
		}

		prose = append(prose, lines[i])
		i++
	}
	flushProse()
	return blocks
}

func renderMarkdownBlocks(blocks []markdownBlock, width int) string {
	if width < 20 {
		width = 20
	}
	var rendered []string
	for _, block := range blocks {
		switch block.kind {
		case markdownBlockProse:
			chunk := renderMarkdownProseBlock(block.text, width)
			if strings.TrimSpace(chunk) != "" {
				rendered = append(rendered, chunk)
			}
		case markdownBlockTable:
			rendered = append(rendered, renderMarkdownTable(block.table, width))
		case markdownBlockCodeFence:
			chunk := renderMarkdownChunk(block.text, width)
			if strings.TrimSpace(chunk) != "" {
				rendered = append(rendered, chunk)
			}
		case markdownBlockDiffFence:
			rendered = append(rendered, renderDiffForView(block.text, width))
		}
	}
	return trimRenderedLines(strings.Join(rendered, "\n"))
}

func renderMarkdownProseBlock(markdown string, width int) string {
	var rendered []string
	var chunk []string
	flushChunk := func() {
		if len(chunk) == 0 {
			return
		}
		text := strings.Join(chunk, "\n")
		chunk = nil
		renderedChunk := renderMarkdownChunk(exposeMarkdownLinks(text), width)
		if strings.TrimSpace(renderedChunk) != "" {
			rendered = append(rendered, renderedChunk)
		}
	}

	for _, line := range strings.Split(markdown, "\n") {
		if heading := markdownHeadingPattern.FindStringSubmatch(line); heading != nil {
			flushChunk()
			for _, wrapped := range wrapChatText(strings.TrimSpace(heading[2]), width) {
				rendered = append(rendered, chatHeaderStyle.Render(wrapped))
			}
			continue
		}
		if markdownHorizontalRulePattern.MatchString(line) {
			flushChunk()
			rendered = append(rendered, chatMetaStyle.Render(strings.Repeat("─", min(width, 24))))
			continue
		}
		chunk = append(chunk, line)
	}
	flushChunk()
	return strings.Join(rendered, "\n")
}

func markdownTableAt(lines []string, start int) (markdownTable, int, bool) {
	if start+1 >= len(lines) {
		return markdownTable{}, start, false
	}
	if markdownLineIsIndentedCode(lines[start]) || markdownLineIsIndentedCode(lines[start+1]) {
		return markdownTable{}, start, false
	}
	headers, ok := parseMarkdownTableRow(lines[start])
	if !ok {
		return markdownTable{}, start, false
	}
	separator, ok := parseMarkdownTableRow(lines[start+1])
	if !ok || !isMarkdownTableSeparator(separator) {
		return markdownTable{}, start, false
	}

	table := markdownTable{headers: headers}
	columns := len(headers)
	i := start + 2
	for ; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "" {
			break
		}
		row, ok := parseMarkdownTableRow(lines[i])
		if !ok || isMarkdownTableSeparator(row) {
			break
		}
		if len(row) > columns {
			columns = len(row)
		}
		table.rows = append(table.rows, row)
	}
	table.headers = padMarkdownTableRow(table.headers, columns)
	for row := range table.rows {
		table.rows[row] = padMarkdownTableRow(table.rows[row], columns)
	}
	return table, i, true
}

func parseMarkdownTableRow(line string) ([]string, bool) {
	trimmed := strings.TrimSpace(line)
	if !strings.Contains(trimmed, "|") {
		return nil, false
	}
	trimmed = strings.TrimPrefix(trimmed, "|")
	trimmed = strings.TrimSuffix(trimmed, "|")

	var cells []string
	var b strings.Builder
	escaped := false
	for _, r := range trimmed {
		switch {
		case escaped:
			if r != '|' {
				b.WriteRune('\\')
			}
			b.WriteRune(r)
			escaped = false
		case r == '\\':
			escaped = true
		case r == '|':
			cells = append(cells, strings.TrimSpace(b.String()))
			b.Reset()
		default:
			b.WriteRune(r)
		}
	}
	if escaped {
		b.WriteRune('\\')
	}
	cells = append(cells, strings.TrimSpace(b.String()))
	if len(cells) < 2 {
		return nil, false
	}
	return cells, true
}

func isMarkdownTableSeparator(cells []string) bool {
	if len(cells) < 2 {
		return false
	}
	for _, cell := range cells {
		normalized := strings.ReplaceAll(strings.TrimSpace(cell), " ", "")
		if !markdownTableSeparatorPattern.MatchString(normalized) {
			return false
		}
	}
	return true
}

func padMarkdownTableRow(row []string, columns int) []string {
	if len(row) >= columns {
		return row
	}
	out := slices.Clone(row)
	for len(out) < columns {
		out = append(out, "")
	}
	return out
}

func renderMarkdownTable(table markdownTable, width int) string {
	if width < 20 {
		width = 20
	}
	rows := make([][]string, 0, len(table.rows)+1)
	rows = append(rows, table.headers)
	rows = append(rows, table.rows...)
	if markdownTableShouldStack(rows, width) {
		return renderMarkdownStackedTable(table, width)
	}

	widths := markdownTableColumnWidths(rows, width)
	var rendered []string
	rendered = append(rendered, renderMarkdownTableRow(table.headers, widths, true)...)
	rendered = append(rendered, strings.Repeat("─", min(width, markdownTableRenderedWidth(widths))))
	for _, row := range table.rows {
		rendered = append(rendered, renderMarkdownTableRow(row, widths, false)...)
	}
	return strings.Join(rendered, "\n")
}

func markdownTableShouldStack(rows [][]string, width int) bool {
	if len(rows) == 0 || len(rows[0]) < 4 {
		return false
	}
	columns := len(rows[0])
	maxWidths := make([]int, columns)
	longCells := 0
	for _, row := range rows {
		for col := 0; col < columns && col < len(row); col++ {
			cell := cleanMarkdownTableCell(row[col])
			cellWidth := lipgloss.Width(cell)
			maxWidths[col] = max(maxWidths[col], cellWidth)
			if cellWidth > max(24, width/3) {
				longCells++
			}
		}
	}
	naturalWidth := 2*(columns-1) + sumInts(maxWidths)
	return naturalWidth > width*2 || longCells >= 2
}

func renderMarkdownStackedTable(table markdownTable, width int) string {
	var rendered []string
	headers := make([]string, len(table.headers))
	for i, header := range table.headers {
		headers[i] = cleanMarkdownTableCell(header)
		if headers[i] == "" {
			headers[i] = "Column " + fmt.Sprint(i+1)
		}
	}

	for rowIndex, row := range table.rows {
		if rowIndex > 0 {
			rendered = append(rendered, "")
		}
		title := ""
		if len(row) > 0 {
			title = cleanMarkdownTableCell(row[0])
		}
		if title != "" {
			for _, line := range wrapChatText(title, width) {
				rendered = append(rendered, chatHeaderStyle.Render(line))
			}
		}
		for col := 1; col < len(headers); col++ {
			value := ""
			if col < len(row) {
				value = cleanMarkdownTableCell(row[col])
			}
			if strings.TrimSpace(value) == "" {
				continue
			}
			rendered = append(rendered, renderMarkdownStackedTableField(headers[col], value, width)...)
		}
		if title == "" {
			value := ""
			if len(row) > 0 {
				value = cleanMarkdownTableCell(row[0])
			}
			if strings.TrimSpace(value) != "" {
				rendered = append(rendered, renderMarkdownStackedTableField(headers[0], value, width)...)
			}
		}
	}
	return strings.Join(rendered, "\n")
}

func renderMarkdownStackedTableField(label, value string, width int) []string {
	prefix := "  " + label + ": "
	bodyWidth := max(10, width-lipgloss.Width(prefix))
	wrapped := wrapTableCell(value, bodyWidth)
	lines := make([]string, 0, len(wrapped))
	for i, line := range wrapped {
		if i == 0 {
			lines = append(lines, chatHeaderStyle.Render("  "+label+": ")+line)
			continue
		}
		lines = append(lines, strings.Repeat(" ", lipgloss.Width(prefix))+line)
	}
	if len(lines) == 0 {
		return []string{chatHeaderStyle.Render("  " + label + ":")}
	}
	return lines
}

func markdownTableColumnWidths(rows [][]string, width int) []int {
	columns := 0
	for _, row := range rows {
		columns = max(columns, len(row))
	}
	if columns == 0 {
		return nil
	}
	gapWidth := 2 * (columns - 1)
	available := max(columns, width-gapWidth)
	minWidth := 4
	if available < columns*minWidth {
		minWidth = max(1, available/columns)
	}

	maxWidths := make([]int, columns)
	for _, row := range rows {
		for col := range columns {
			cell := ""
			if col < len(row) {
				cell = cleanMarkdownTableCell(row[col])
			}
			for _, line := range strings.Split(cell, "\n") {
				maxWidths[col] = max(maxWidths[col], lipgloss.Width(line))
			}
		}
	}

	widths := make([]int, columns)
	even := max(minWidth, available/columns)
	for col := range widths {
		widths[col] = min(max(maxWidths[col], minWidth), even)
	}

	for leftover := available - sumInts(widths); leftover > 0; {
		grew := false
		for col := columns - 1; col >= 0 && leftover > 0; col-- {
			if widths[col] >= maxWidths[col] {
				continue
			}
			widths[col]++
			leftover--
			grew = true
		}
		if !grew {
			widths[columns-1] += leftover
			break
		}
	}

	for sumInts(widths) > available {
		col := widestColumn(widths)
		if widths[col] <= minWidth {
			break
		}
		widths[col]--
	}
	return widths
}

func renderMarkdownTableRow(row []string, widths []int, header bool) []string {
	wrapped := make([][]string, len(widths))
	height := 1
	for col := range widths {
		cell := ""
		if col < len(row) {
			cell = cleanMarkdownTableCell(row[col])
		}
		wrapped[col] = wrapTableCell(cell, widths[col])
		height = max(height, len(wrapped[col]))
	}

	lines := make([]string, 0, height)
	for lineIndex := range height {
		var b strings.Builder
		for col := range widths {
			if col > 0 {
				b.WriteString("  ")
			}
			part := ""
			if lineIndex < len(wrapped[col]) {
				part = wrapped[col][lineIndex]
			}
			if header || (col == 0 && strings.TrimSpace(part) != "") {
				part = chatHeaderStyle.Render(part)
			}
			if col < len(widths)-1 {
				part = padRenderedCell(part, widths[col])
			}
			b.WriteString(part)
		}
		lines = append(lines, strings.TrimRight(b.String(), " "))
	}
	return lines
}

func cleanMarkdownTableCell(cell string) string {
	cell = strings.TrimSpace(cell)
	cell = strings.ReplaceAll(cell, "<br>", "\n")
	cell = strings.ReplaceAll(cell, "<br/>", "\n")
	cell = strings.ReplaceAll(cell, "<br />", "\n")
	cell = exposeMarkdownLinks(cell)
	cell = strings.ReplaceAll(cell, "**", "")
	cell = strings.ReplaceAll(cell, "__", "")
	cell = strings.ReplaceAll(cell, "`", "")
	return strings.ReplaceAll(cell, `\|`, "|")
}

func wrapTableCell(text string, width int) []string {
	if width <= 0 {
		return []string{""}
	}
	var out []string
	for _, rawLine := range strings.Split(text, "\n") {
		line := strings.TrimRight(rawLine, "\r")
		for lipgloss.Width(line) > width {
			runes := []rune(line)
			cut := tableCellWrapCut(runes, width)
			out = append(out, strings.TrimSpace(string(runes[:cut])))
			line = strings.TrimSpace(string(runes[cut:]))
		}
		out = append(out, line)
	}
	if len(out) == 0 {
		return []string{""}
	}
	return out
}

func tableCellWrapCut(runes []rune, width int) int {
	if len(runes) == 0 {
		return 0
	}
	cut := 0
	lineWidth := 0
	for i, r := range runes {
		nextWidth := lipgloss.Width(string(r))
		if cut > 0 && lineWidth+nextWidth > width {
			break
		}
		lineWidth += nextWidth
		cut = i + 1
	}
	if cut <= 0 {
		return 1
	}
	preferred := cut
	spaceWidth := 0
	for i := 0; i < cut; i++ {
		spaceWidth += lipgloss.Width(string(runes[i]))
		if unicode.IsSpace(runes[i]) && spaceWidth >= max(1, width/2) {
			preferred = i + 1
		}
	}
	return preferred
}

func markdownTableRenderedWidth(widths []int) int {
	if len(widths) == 0 {
		return 0
	}
	return sumInts(widths) + 2*(len(widths)-1)
}

func padRenderedCell(cell string, width int) string {
	padding := width - lipgloss.Width(cell)
	if padding <= 0 {
		return cell
	}
	return cell + strings.Repeat(" ", padding)
}

func sumInts(values []int) int {
	total := 0
	for _, value := range values {
		total += value
	}
	return total
}

func widestColumn(widths []int) int {
	widest := 0
	for i := 1; i < len(widths); i++ {
		if widths[i] > widths[widest] {
			widest = i
		}
	}
	return widest
}

func markdownRendererForWidth(width int) (*cachedMarkdownRenderer, error) {
	return chatMarkdownRenderers.renderer(width)
}

func (c *markdownRendererCache) renderer(width int) (*cachedMarkdownRenderer, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if cached, ok := c.renderers[width]; ok {
		return cached, nil
	}
	renderer, err := glamour.NewTermRenderer(
		glamour.WithStyles(compactMarkdownStyle()),
		glamour.WithWordWrap(width),
		glamour.WithTableWrap(true),
		glamour.WithInlineTableLinks(true),
	)
	if err != nil {
		return nil, err
	}
	cached := &cachedMarkdownRenderer{renderer: renderer}
	if len(c.order) >= maxMarkdownRendererCacheEntries {
		evict := c.order[0]
		c.order = c.order[1:]
		delete(c.renderers, evict)
	}
	c.renderers[width] = cached
	c.order = append(c.order, width)
	return cached, nil
}

func (c *markdownRendererCache) len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.renderers)
}

func markdownFenceStart(line string) (string, string, bool) {
	trimmed := strings.TrimSpace(line)
	if strings.HasPrefix(trimmed, "```") {
		return markdownFence(trimmed, '`')
	}
	if strings.HasPrefix(trimmed, "~~~") {
		return markdownFence(trimmed, '~')
	}
	return "", "", false
}

func markdownFence(line string, marker rune) (string, string, bool) {
	count := 0
	for _, r := range line {
		if r != marker {
			break
		}
		count++
	}
	if count < 3 {
		return "", "", false
	}
	info := strings.TrimSpace(line[count:])
	return strings.Repeat(string(marker), count), info, true
}

func markdownFenceIsDiff(info string) bool {
	fields := strings.Fields(strings.ToLower(strings.TrimSpace(info)))
	if len(fields) == 0 {
		return false
	}
	first := strings.Trim(fields[0], "{}.")
	return first == "diff" || first == "patch"
}

func fenceEnd(line string, fence string) bool {
	trimmed := strings.TrimSpace(line)
	return strings.HasPrefix(trimmed, fence)
}

func markdownLineIsIndentedCode(line string) bool {
	if strings.HasPrefix(line, "\t") {
		return true
	}
	spaces := 0
	for _, r := range line {
		if r != ' ' {
			break
		}
		spaces++
	}
	return spaces >= 4
}

func compactMarkdownStyle() glamouransi.StyleConfig {
	palette := markdownANSIPalette()
	style := styles.ASCIIStyleConfig
	style.Document.Margin = uintPtr(0)
	style.Document.BlockPrefix = ""
	style.Document.BlockSuffix = ""
	style.BlockQuote.Color = optionalStringPtr(palette.muted)
	style.CodeBlock.Margin = uintPtr(0)
	style.CodeBlock.Color = optionalStringPtr(palette.code)
	style.Table.Margin = uintPtr(0)
	style.Table.Color = optionalStringPtr(palette.table)
	style.Heading.Color = optionalStringPtr(palette.heading)
	style.Heading.Bold = boolPtr(true)
	style.H1.Prefix = ""
	style.H2.Prefix = ""
	style.H3.Prefix = ""
	style.H4.Prefix = ""
	style.H5.Prefix = ""
	style.H6.Prefix = ""
	style.Strong.BlockPrefix = ""
	style.Strong.BlockSuffix = ""
	style.Strong.Bold = boolPtr(true)
	style.Strong.Color = optionalStringPtr(palette.strong)
	style.Emph.BlockPrefix = ""
	style.Emph.BlockSuffix = ""
	style.Emph.Italic = boolPtr(true)
	style.Emph.Color = optionalStringPtr(palette.muted)
	style.Code.BlockPrefix = ""
	style.Code.BlockSuffix = ""
	style.Code.Color = optionalStringPtr(palette.code)
	style.Link.Color = optionalStringPtr(palette.link)
	style.LinkText.Color = optionalStringPtr(palette.link)
	style.LinkText.Underline = boolPtr(true)
	style.HorizontalRule.Color = optionalStringPtr(palette.muted)
	style.HorizontalRule.Format = "\n" + strings.Repeat("─", 24) + "\n"
	return style
}

type markdownTerminalPalette struct {
	heading string
	strong  string
	link    string
	code    string
	muted   string
	table   string
}

func markdownPaletteForBackground(dark bool) markdownTerminalPalette {
	return markdownANSIPalette()
}

func markdownANSIPalette() markdownTerminalPalette {
	return markdownTerminalPalette{
		link:  chatAnsiBlue,
		code:  chatAnsiCyan,
		muted: chatAnsiMuted,
	}
}

func trimRenderedLines(rendered string) string {
	rendered = strings.TrimRight(rendered, "\n")
	lines := strings.Split(rendered, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimRight(line, " \t")
	}
	return strings.Join(lines, "\n")
}

func uintPtr(value uint) *uint {
	return &value
}

func boolPtr(value bool) *bool {
	return &value
}

func optionalStringPtr(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

func splitRenderedBody(body string) []string {
	body = strings.TrimRight(body, "\n")
	if body == "" {
		return []string{""}
	}
	return strings.Split(body, "\n")
}
