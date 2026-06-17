package tui

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

	chatAssistantStyle = lipgloss.NewStyle()

	chatThinkingStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(chatAnsiMuted)).
				Italic(true)

	chatToolStyle = lipgloss.NewStyle()

	chatToolRunningStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(chatAnsiYellow))

	chatToolDoneStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(chatAnsiGreen))

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
				Bold(true).
				Foreground(lipgloss.Color(chatAnsiRed))

	chatInputStyle = lipgloss.NewStyle()

	chatInputBorderStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(chatAnsiMuted))

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
