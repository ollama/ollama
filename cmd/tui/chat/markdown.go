package chat

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/charmbracelet/lipgloss"
	"github.com/mattn/go-runewidth"
)

func renderMarkdownForView(markdown string, width int) string {
	if width < 20 {
		width = 20
	}

	source := strings.Split(strings.TrimRight(markdown, "\n"), "\n")
	var rendered []string
	inCodeBlock := false
	codeLanguage := ""
	for i := 0; i < len(source); i++ {
		line := strings.TrimRight(source[i], "\r")
		trimmed := strings.TrimSpace(line)

		if strings.HasPrefix(trimmed, "```") {
			inCodeBlock = !inCodeBlock
			if inCodeBlock {
				codeLanguage = markdownCodeFenceLanguage(trimmed)
			} else {
				codeLanguage = ""
			}
			continue
		}
		if inCodeBlock {
			rendered = append(rendered, renderMarkdownCodeLine(line, codeLanguage, width)...)
			continue
		}

		if table, consumed := renderMarkdownTable(source[i:], width); consumed > 0 {
			rendered = append(rendered, table...)
			i += consumed - 1
			continue
		}

		if heading, ok := markdownHeading(trimmed); ok {
			rendered = append(rendered, chatHeaderStyle.Render(renderMarkdownRunes(parseMarkdownInline(heading))))
			continue
		}

		if trimmed == "" {
			rendered = append(rendered, "")
			continue
		}
		rendered = append(rendered, wrapMarkdownInline(line, width)...)
	}
	return strings.Join(rendered, "\n")
}

func splitRenderedBody(body string) []string {
	body = strings.TrimRight(body, "\n")
	if body == "" {
		return []string{""}
	}
	return strings.Split(body, "\n")
}

func markdownHeading(line string) (string, bool) {
	if !strings.HasPrefix(line, "#") {
		return "", false
	}
	level := 0
	for level < len(line) && line[level] == '#' {
		level++
	}
	if level == 0 || level > 6 || level >= len(line) || line[level] != ' ' {
		return "", false
	}
	return strings.TrimSpace(line[level:]), true
}

type markdownInlineStyle uint8

const (
	markdownPlain markdownInlineStyle = iota
	markdownStrong
	markdownCode
)

type markdownInlineRune struct {
	r     rune
	style markdownInlineStyle
}

// wrapMarkdownInline parses a complete source line before wrapping it. That
// keeps emphasis intact when its opening and closing delimiters land on
// different visual lines.
func wrapMarkdownInline(line string, width int) []string {
	return wrapInlineRunes(parseMarkdownInline(line), width)
}

func wrapInlineRunes(runes []markdownInlineRune, width int) []string {
	if len(runes) == 0 {
		return []string{""}
	}

	var rendered []string
	for len(runes) > 0 {
		hardCut, spaceCut, currentWidth := 0, 0, 0
		for i, item := range runes {
			nextWidth := currentWidth + runewidth.RuneWidth(item.r)
			if nextWidth > width {
				break
			}
			currentWidth = nextWidth
			hardCut = i + 1
			if unicode.IsSpace(item.r) && currentWidth > width/2 {
				spaceCut = i
			}
		}
		cut := hardCut
		if spaceCut > 0 {
			cut = spaceCut
		}
		if cut == 0 {
			cut = 1
		}

		lineRunes := trimMarkdownSpace(runes[:cut])
		rendered = append(rendered, renderMarkdownRunes(lineRunes))
		runes = trimMarkdownSpace(runes[cut:])
	}
	return rendered
}

func parseMarkdownInline(line string) []markdownInlineRune {
	var out []markdownInlineRune
	for len(line) > 0 {
		if strings.HasPrefix(line, "`") {
			if end := strings.Index(line[1:], "`"); end >= 0 {
				out = appendMarkdownRunes(out, line[1:end+1], markdownCode)
				line = line[end+2:]
				continue
			}
		}
		if (strings.HasPrefix(line, "**") || strings.HasPrefix(line, "__")) && canOpenMarkdownStrong(out) {
			delimiter := line[:2]
			if end := strings.Index(line[2:], delimiter); end >= 0 {
				out = appendMarkdownRunes(out, line[2:end+2], markdownStrong)
				line = line[end+4:]
				continue
			}
		}

		r, size := utf8.DecodeRuneInString(line)
		out = append(out, markdownInlineRune{r: r, style: markdownPlain})
		line = line[size:]
	}
	return out
}

// canOpenMarkdownStrong keeps delimiter-like text in bare URLs and identifiers
// literal, only treating ** / __ as strong emphasis at the common
// whitespace- or punctuation-delimited form.
func canOpenMarkdownStrong(out []markdownInlineRune) bool {
	if len(out) == 0 {
		return true
	}
	previous := out[len(out)-1].r
	return (unicode.IsSpace(previous) || unicode.IsPunct(previous)) && !markdownStrongInURL(out)
}

func markdownStrongInURL(out []markdownInlineRune) bool {
	start := len(out)
	for start > 0 && !unicode.IsSpace(out[start-1].r) {
		start--
	}

	var token strings.Builder
	for _, item := range out[start:] {
		token.WriteRune(item.r)
	}
	return strings.Contains(token.String(), "://")
}

func appendMarkdownRunes(out []markdownInlineRune, text string, style markdownInlineStyle) []markdownInlineRune {
	for _, r := range text {
		out = append(out, markdownInlineRune{r: r, style: style})
	}
	return out
}

func trimMarkdownSpace(runes []markdownInlineRune) []markdownInlineRune {
	start, end := 0, len(runes)
	for start < end && unicode.IsSpace(runes[start].r) {
		start++
	}
	for end > start && unicode.IsSpace(runes[end-1].r) {
		end--
	}
	return runes[start:end]
}

func renderMarkdownRunes(runes []markdownInlineRune) string {
	var b strings.Builder
	for start := 0; start < len(runes); {
		end := start + 1
		for end < len(runes) && runes[end].style == runes[start].style {
			end++
		}
		var text strings.Builder
		for _, item := range runes[start:end] {
			text.WriteRune(item.r)
		}
		switch runes[start].style {
		case markdownStrong:
			b.WriteString(chatStrongStyle.Render(text.String()))
		case markdownCode:
			b.WriteString(chatInlineCodeStyle.Render(text.String()))
		default:
			b.WriteString(text.String())
		}
		start = end
	}
	return b.String()
}

func renderMarkdownCodeLine(line, language string, width int) []string {
	codeWidth := max(1, width-2)
	lines := wrapChatText(line, codeWidth)
	for i, wrapped := range lines {
		lines[i] = "  " + chatCodeBlockStyle.Render(highlightMarkdownCodeLine(language, wrapped))
	}
	return lines
}

func markdownCodeFenceLanguage(fence string) string {
	info := strings.TrimSpace(strings.TrimPrefix(fence, "```"))
	if info == "" {
		return ""
	}
	return normalizeMarkdownCodeLanguage(strings.Fields(info)[0])
}

func normalizeMarkdownCodeLanguage(language string) string {
	switch strings.ToLower(strings.TrimSpace(language)) {
	case "go", "golang":
		return "go"
	case "js", "javascript", "jsx", "ts", "typescript", "tsx":
		return "javascript"
	case "py", "python":
		return "python"
	case "sh", "shell", "bash", "zsh":
		return "shell"
	case "json":
		return "json"
	case "yaml", "yml":
		return "yaml"
	case "sql":
		return "sql"
	case "c", "h", "cpp", "c++", "cc", "cxx", "hpp", "java", "rust", "rs", "css", "swift":
		return "clike"
	default:
		return ""
	}
}

func highlightMarkdownCodeLine(language, line string) string {
	if language == "" || line == "" {
		return line
	}

	var rendered strings.Builder
	for i := 0; i < len(line); {
		if comment, ok := markdownCodeComment(language, line[i:]); ok {
			rendered.WriteString(chatCodeCommentStyle.Render(comment))
			break
		}
		if quote := line[i]; quote == '\'' || quote == '"' || (quote == '`' && language != "json") {
			end := markdownCodeStringEnd(line, i, quote)
			rendered.WriteString(chatCodeStringStyle.Render(line[i:end]))
			i = end
			continue
		}
		if isMarkdownCodeIdentifierStart(line[i]) {
			end := i + 1
			for end < len(line) && isMarkdownCodeIdentifierPart(line[end]) {
				end++
			}
			word := line[i:end]
			rendered.WriteString(renderMarkdownCodeToken(markdownCodeTokenKindFor(language, word, line[end:]), word))
			i = end
			continue
		}
		if line[i] >= '0' && line[i] <= '9' {
			end := i + 1
			for end < len(line) && isMarkdownCodeNumberPart(line[end]) {
				end++
			}
			rendered.WriteString(chatCodeNumberStyle.Render(line[i:end]))
			i = end
			continue
		}
		rendered.WriteByte(line[i])
		i++
	}
	return rendered.String()
}

func markdownCodeComment(language, line string) (string, bool) {
	if strings.HasPrefix(line, "//") && language != "shell" && language != "yaml" {
		return line, true
	}
	if strings.HasPrefix(line, "#") && (language == "shell" || language == "python" || language == "yaml") {
		return line, true
	}
	if strings.HasPrefix(line, "--") && language == "sql" {
		return line, true
	}
	if strings.HasPrefix(line, "/*") {
		return line, true
	}
	return "", false
}

func markdownCodeStringEnd(line string, start int, quote byte) int {
	for i := start + 1; i < len(line); i++ {
		if line[i] == '\\' {
			i++
			continue
		}
		if line[i] == quote {
			return i + 1
		}
	}
	return len(line)
}

func isMarkdownCodeIdentifierStart(c byte) bool {
	return c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
}

func isMarkdownCodeIdentifierPart(c byte) bool {
	return isMarkdownCodeIdentifierStart(c) || c >= '0' && c <= '9'
}

func isMarkdownCodeNumberPart(c byte) bool {
	return c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F' || c == '.' || c == '_' || c == 'x' || c == 'X'
}

func markdownCodeFunctionCall(rest string) bool {
	return strings.HasPrefix(strings.TrimLeft(rest, " \t"), "(")
}

type markdownCodeTokenKind uint8

const (
	markdownCodeTokenPlain markdownCodeTokenKind = iota
	markdownCodeTokenKeyword
	markdownCodeTokenLiteral
	markdownCodeTokenType
	markdownCodeTokenFunction
)

func markdownCodeTokenKindFor(language, word, rest string) markdownCodeTokenKind {
	switch {
	case markdownCodeKeywords[language][word], markdownCodeKeywords["all"][word]:
		return markdownCodeTokenKeyword
	case markdownCodeLiterals[word]:
		return markdownCodeTokenLiteral
	case markdownCodeTypes[word]:
		return markdownCodeTokenType
	case markdownCodeFunctionCall(rest):
		return markdownCodeTokenFunction
	default:
		return markdownCodeTokenPlain
	}
}

func renderMarkdownCodeToken(kind markdownCodeTokenKind, text string) string {
	switch kind {
	case markdownCodeTokenKeyword:
		return chatCodeKeywordStyle.Render(text)
	case markdownCodeTokenLiteral:
		return chatCodeNumberStyle.Render(text)
	case markdownCodeTokenType:
		return chatCodeTypeStyle.Render(text)
	case markdownCodeTokenFunction:
		return chatCodeFunctionStyle.Render(text)
	default:
		return text
	}
}

var markdownCodeKeywords = map[string]map[string]bool{
	"all": {
		"break": true, "case": true, "catch": true, "class": true, "const": true, "continue": true, "default": true, "defer": true, "do": true, "else": true, "for": true, "func": true, "function": true, "if": true, "import": true, "in": true, "let": true, "new": true, "package": true, "private": true, "protected": true, "public": true, "return": true, "struct": true, "switch": true, "throw": true, "try": true, "var": true, "while": true,
	},
	"go":         {"chan": true, "go": true, "map": true, "range": true, "select": true, "type": true},
	"javascript": {"async": true, "await": true, "export": true, "extends": true, "from": true, "interface": true, "of": true, "static": true, "typeof": true},
	"python":     {"and": true, "as": true, "def": true, "del": true, "elif": true, "except": true, "finally": true, "global": true, "is": true, "lambda": true, "nonlocal": true, "not": true, "or": true, "pass": true, "raise": true, "with": true, "yield": true},
	"shell":      {"done": true, "echo": true, "esac": true, "export": true, "fi": true, "local": true, "then": true},
	"clike":      {"enum": true, "implements": true, "namespace": true, "template": true, "using": true},
	"sql":        {"delete": true, "from": true, "insert": true, "into": true, "join": true, "select": true, "update": true, "where": true},
	"json":       {},
	"yaml":       {},
}

var markdownCodeLiterals = map[string]bool{
	"false": true, "nil": true, "none": true, "null": true, "true": true, "undefined": true,
}

var markdownCodeTypes = map[string]bool{
	"any": true, "bool": true, "boolean": true, "byte": true, "error": true, "float": true, "float32": true, "float64": true, "int": true, "int8": true, "int16": true, "int32": true, "int64": true, "number": true, "object": true, "rune": true, "string": true, "uint": true, "uint8": true, "uint16": true, "uint32": true, "uint64": true, "void": true,
}

func renderMarkdownTable(lines []string, width int) ([]string, int) {
	if len(lines) < 2 || !looksLikeMarkdownTableRow(lines[0]) || !isMarkdownTableSeparator(lines[1]) {
		return nil, 0
	}

	var rows [][]string
	consumed := 0
	for consumed < len(lines) && looksLikeMarkdownTableRow(lines[consumed]) {
		if consumed == 1 && isMarkdownTableSeparator(lines[consumed]) {
			consumed++
			continue
		}
		rows = append(rows, parseMarkdownTableRow(lines[consumed]))
		consumed++
	}
	if len(rows) == 0 {
		return nil, 0
	}

	columnCount := 0
	for _, row := range rows {
		columnCount = max(columnCount, len(row))
	}
	naturalWidths := make([]int, columnCount)
	for _, row := range rows {
		for i := range columnCount {
			cell := ""
			if i < len(row) {
				cell = row[i]
			}
			naturalWidths[i] = max(naturalWidths[i], markdownInlineWidth(cell))
		}
	}
	widths := markdownTableColumnWidths(naturalWidths, width)

	var rendered []string
	for rowIndex, row := range rows {
		wrappedCells := make([][]string, columnCount)
		rowHeight := 1
		for i := range columnCount {
			cell := ""
			if i < len(row) {
				cell = row[i]
			}
			wrappedCells[i] = wrapMarkdownTableCell(cell, widths[i])
			rowHeight = max(rowHeight, len(wrappedCells[i]))
		}
		for lineIndex := range rowHeight {
			cells := make([]string, columnCount)
			for i := range columnCount {
				cellLine := ""
				if lineIndex < len(wrappedCells[i]) {
					cellLine = wrappedCells[i][lineIndex]
				}
				cells[i] = padPlainLine(cellLine, widths[i])
			}
			line := strings.Join(cells, chatTableBorderStyle.Render(" | "))
			if rowIndex == 0 {
				line = chatHeaderStyle.Render(stripANSIForWidth(line))
			}
			rendered = append(rendered, line)
		}
	}
	return rendered, consumed
}

func markdownTableColumnWidths(naturalWidths []int, width int) []int {
	if len(naturalWidths) == 0 {
		return nil
	}
	separatorWidth := max(0, len(naturalWidths)-1) * lipglossWidth(" | ")
	available := max(1, width-separatorWidth)
	widths := make([]int, len(naturalWidths))
	minWidths := make([]int, len(naturalWidths))
	for i, natural := range naturalWidths {
		widths[i] = max(1, natural)
		minWidth := min(widths[i], 12)
		if i == 0 {
			minWidth = min(widths[i], 4)
		}
		minWidths[i] = max(1, minWidth)
	}

	for sumInts(widths) > available {
		index := widestShrinkableColumn(widths, minWidths)
		if index < 0 {
			break
		}
		widths[index]--
	}
	for sumInts(widths) > available {
		index := widestColumn(widths)
		if index < 0 || widths[index] <= 1 {
			break
		}
		widths[index]--
	}
	return widths
}

func widestShrinkableColumn(widths, minWidths []int) int {
	index := -1
	for i, width := range widths {
		if width <= minWidths[i] {
			continue
		}
		if index < 0 || width > widths[index] {
			index = i
		}
	}
	return index
}

func widestColumn(widths []int) int {
	index := -1
	for i, width := range widths {
		if index < 0 || width > widths[index] {
			index = i
		}
	}
	return index
}

func sumInts(values []int) int {
	sum := 0
	for _, value := range values {
		sum += value
	}
	return sum
}

func wrapMarkdownTableCell(cell string, width int) []string {
	lines := wrapInlineRunes(parseMarkdownInline(cell), max(1, width))
	if len(lines) == 0 {
		return []string{""}
	}
	return lines
}

// markdownInlineWidth reports the visible width of a cell once Markdown
// delimiters are parsed away, so columns size to rendered content.
func markdownInlineWidth(cell string) int {
	width := 0
	for _, item := range parseMarkdownInline(cell) {
		width += runewidth.RuneWidth(item.r)
	}
	return width
}

func looksLikeMarkdownTableRow(line string) bool {
	line = strings.TrimSpace(line)
	return strings.Contains(line, "|") && strings.Count(line, "|") >= 1
}

func isMarkdownTableSeparator(line string) bool {
	cells := parseMarkdownTableRow(line)
	if len(cells) == 0 {
		return false
	}
	for _, cell := range cells {
		cell = strings.Trim(cell, " :-")
		if cell != "" {
			return false
		}
	}
	return true
}

func parseMarkdownTableRow(line string) []string {
	line = strings.TrimSpace(line)
	line = strings.TrimPrefix(line, "|")
	line = strings.TrimSuffix(line, "|")
	raw := strings.Split(line, "|")
	cells := make([]string, 0, len(raw))
	for _, cell := range raw {
		cells = append(cells, strings.TrimSpace(cell))
	}
	return cells
}

func padPlainLine(line string, width int) string {
	if extra := width - lipglossWidth(line); extra > 0 {
		return line + strings.Repeat(" ", extra)
	}
	return line
}

func stripANSIForWidth(line string) string {
	return stripChatANSI(line)
}

func lipglossWidth(line string) int {
	return lipgloss.Width(line)
}
