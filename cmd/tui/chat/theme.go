package chat

import "github.com/charmbracelet/lipgloss"

const (
	chatAnsiRed    = "1"
	chatAnsiGreen  = "2"
	chatAnsiYellow = "3"
	chatAnsiBlue   = "4"
	chatAnsiCyan   = "6"
	chatAnsiMuted  = "8"
)

var (
	chatHeaderStyle = lipgloss.NewStyle().
			Bold(true)

	chatMetaStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color(chatAnsiMuted))

	chatNotificationStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("7"))

	chatUserStyle = lipgloss.NewStyle()

	chatUserBlockStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#111111")).
				Background(lipgloss.Color("#eeeeee"))

	chatAssistantStyle = lipgloss.NewStyle()

	chatToolStyle = lipgloss.NewStyle()

	chatInlineCodeStyle = lipgloss.NewStyle().
				Background(lipgloss.Color(chatAnsiMuted))

	chatCodeBlockStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("7"))

	chatTableBorderStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(chatAnsiMuted))

	chatToolRunningStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(chatAnsiMuted))

	chatToolDoneStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(chatAnsiMuted))

	// chatToolMixedStyle marks a tool group with both succeeded and failed
	// calls (partial success). Amber/orange is distinct from green (success),
	// red (failure), and yellow (running).
	chatToolMixedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("208"))

	chatDiffMetaStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(chatAnsiMuted))

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
				Foreground(lipgloss.Color("#9f5f5f"))

	chatCommandNameStyle = lipgloss.NewStyle()

	chatResumeTextStyle = lipgloss.NewStyle()

	chatResumeTitleStyle = lipgloss.NewStyle().
				Bold(true)

	chatResumeSelectedStyle = lipgloss.NewStyle().
				Bold(true)

	chatResumeMetaStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(chatAnsiMuted))

	chatResumeBorderStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(chatAnsiMuted))

	chatHistoryTitleStyle = lipgloss.NewStyle().
				Bold(true)

	chatHistorySystemRoleStyle = lipgloss.NewStyle().
					Bold(true).
					Foreground(lipgloss.Color(chatAnsiMuted))

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
				Foreground(lipgloss.Color(chatAnsiMuted))

	chatHistoryTextStyle = lipgloss.NewStyle()

	chatHistoryCodeStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(chatAnsiCyan))

	chatSelectionStyle = lipgloss.NewStyle().
				Reverse(true)
)
