package chat

import "github.com/charmbracelet/lipgloss"

const (
	chatAnsiRed         = "1"
	chatAnsiGreen       = "2"
	chatAnsiYellow      = "3"
	chatAnsiBlue        = "4"
	chatAnsiCyan        = "6"
	chatAnsiBrightBlack = "8"
)

var (
	chatHeaderStyle = lipgloss.NewStyle().
			Bold(true)

	chatMetaStyle = lipgloss.NewStyle().
			Faint(true)

	chatFooterStyle = lipgloss.NewStyle().
			Faint(true)

	chatInputBorderStyle = lipgloss.NewStyle().
				Faint(true)

	chatInputPlaceholderStyle = lipgloss.NewStyle().
					Foreground(lipgloss.Color("8"))

	chatCursorStyle = lipgloss.NewStyle().
			Reverse(true)

	chatBlankCursorStyle = lipgloss.NewStyle().
				Faint(true)

	chatNotificationStyle = chatMetaStyle

	chatUserStyle = lipgloss.NewStyle()

	chatUserBlockStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "#777777", Dark: "#8a8a8a"})

	chatToolStyle = lipgloss.NewStyle()

	chatInlineCodeStyle = lipgloss.NewStyle().
				Bold(true)

	chatStrongStyle = lipgloss.NewStyle().
			Bold(true)

	chatCodeBlockStyle = lipgloss.NewStyle()

	chatCodeCommentStyle = lipgloss.NewStyle().
				Faint(true).
				Foreground(lipgloss.AdaptiveColor{Light: "#6a737d", Dark: "#8b949e"})

	chatCodeKeywordStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "#a626a4", Dark: "#c678dd"})

	chatCodeStringStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "#50a14f", Dark: "#98c379"})

	chatCodeNumberStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "#c18401", Dark: "#d19a66"})

	chatCodeFunctionStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "#4078f2", Dark: "#61afef"})

	chatCodeTypeStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "#e45649", Dark: "#e06c75"})

	chatTableBorderStyle = lipgloss.NewStyle().
				Faint(true)

	chatToolRunningStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(chatAnsiYellow))

	chatToolDoneStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(chatAnsiGreen))

	// chatToolMixedStyle marks a tool group with both succeeded and failed
	// calls (partial success). Amber/orange is distinct from green (success),
	// red (failure), and yellow (running).
	chatToolMixedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("208"))

	chatToolOutputStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "#666666", Dark: "#a0a0a0"})

	chatDiffMetaStyle = lipgloss.NewStyle().
				Faint(true)

	chatDiffFileStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color(chatAnsiCyan))

	chatDiffHunkStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(chatAnsiBlue))

	chatDiffAddStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(chatAnsiGreen))

	chatDiffDeleteStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(chatAnsiRed))

	chatErrorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color(chatAnsiRed))

	chatFullAccessStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "#9f5f5f", Dark: "#b87373"})

	chatCommandNameStyle = lipgloss.NewStyle()

	chatPickerTextStyle = lipgloss.NewStyle()

	chatPickerTitleStyle = lipgloss.NewStyle().
				Bold(true)

	chatPickerSelectedStyle = lipgloss.NewStyle().
				Bold(true)

	chatPickerMetaStyle = lipgloss.NewStyle().
				Faint(true)

	chatHistoryTitleStyle = lipgloss.NewStyle().
				Bold(true)

	chatHistorySystemRoleStyle = lipgloss.NewStyle().
					Bold(true).
					Faint(true)

	chatHistoryUserRoleStyle = lipgloss.NewStyle().
					Bold(true).
					Foreground(lipgloss.Color(chatAnsiBlue))

	chatHistoryAssistantRoleStyle = lipgloss.NewStyle().
					Bold(true).
					Foreground(lipgloss.Color(chatAnsiYellow))

	chatHistoryToolRoleStyle = lipgloss.NewStyle().
					Bold(true).
					Foreground(lipgloss.Color(chatAnsiGreen))

	chatHistoryLabelStyle = lipgloss.NewStyle().
				Faint(true)

	chatHistoryTextStyle = lipgloss.NewStyle()
)
