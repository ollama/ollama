package main

import (
	"context"
	"fmt"
	"image/color"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/data/binding"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"
)

// Constants
const (
	modelName   = "llama3.1"
	httpTimeout = 30 * time.Second
	appIconPath = "../app/assets/app.ico"
)

var (
	myApp         fyne.App
	myWindow      fyne.Window
	chatData      binding.StringList
	scroll        *container.Scroll
	savedChats    []string
	currentChatID int
	mu            sync.Mutex
)

func main() {
	initializeApp()
}

// loadChatHistory loads the chat history for the selected saved chat
func loadChatHistory() {
	existingItems, _ := chatData.Get()
	newItems := []string{
		fmt.Sprintf("system: Chat %d loaded.", currentChatID+1),
		"assistant: How can I help you?",
	}
	chatData.Set(append(existingItems, newItems...))
}

// makeMainUI creates the primary UI layout
func makeMainUI() fyne.CanvasObject {
	// Chat history
	chatHistory := createChatHistory()

	// Input components (message input field and send button container)
	messageInput, sendButton := createInputComponents()

	// Input container (aligns the input field and send button)
	inputContainer := container.NewBorder(nil, nil, nil, sendButton, messageInput)

	// Chat pane (combines chat history and input container at the bottom)
	messagePane := container.NewBorder(nil, inputContainer, nil, nil, chatHistory)

	// Server list (saved chats list)
	serverList := widget.NewList(
		func() int { return len(savedChats) },
		func() fyne.CanvasObject { return widget.NewLabel("") },
		func(id widget.ListItemID, o fyne.CanvasObject) {
			o.(*widget.Label).SetText(savedChats[id])
		},
	)

	// Handle switching between saved chats
	serverList.OnSelected = func(id widget.ListItemID) {
		currentChatID = id
		loadChatHistory() // Load the chat history for the selected chat
	}

	// Combine the server list (left) and message pane (right) in a horizontal split
	mainContent := container.NewHSplit(serverList, messagePane)
	mainContent.SetOffset(0.2) // Set the initial split ratio (20% server list, 80% chat pane)

	return mainContent
}

// setTheme toggles between light and dark themes
func setTheme(isDark bool) {
	if isDark {
		myApp.Settings().SetTheme(theme.DarkTheme())
	} else {
		myApp.Settings().SetTheme(theme.LightTheme())
	}
	// rebuildChatHistory()
}

func rebuildChatHistory() {
	chatContent := scroll.Content.(*fyne.Container)

	chatContent.Objects = nil

	items, _ := chatData.Get()
	for _, message := range items {
		role, content := parseRoleAndContent(message)
		isUser := (role == "user")

		// Only display user and assistant messages in bubbles
		if role == "user" || role == "assistant" {
			chatContent.Add(createChatBubble(content, isUser))
		}
	}

	chatContent.Refresh()
}

// parseRoleAndContent splits a string like "assistant: Hello" into ("assistant", "Hello")
func parseRoleAndContent(line string) (string, string) {
	parts := strings.SplitN(line, ":", 2)
	if len(parts) != 2 {
		// If somehow not formatted correctly, treat entire line as 'system'
		return "system", line
	}

	role := strings.TrimSpace(parts[0])
	content := strings.TrimSpace(parts[1])
	return role, content
}

// loadAppIcon loads the application icon
func loadAppIcon(relativePath string) fyne.Resource {
	absPath, err := filepath.Abs(relativePath)
	if err != nil {
		fmt.Printf("Failed to resolve app icon path: %v\n", err)
		return nil
	}

	iconData, err := os.ReadFile(absPath)
	if err != nil {
		fmt.Printf("Failed to load app icon: %v\n", err)
		return nil
	}
	return fyne.NewStaticResource("app.ico", iconData)
}

func initializeApp() {
	myApp = app.NewWithID("ollama.gui")
	myWindow = myApp.NewWindow("Ollama GUI")

	// Set app icon
	appIcon := loadAppIcon(appIconPath)
	if appIcon != nil {
		myWindow.SetIcon(appIcon)
	}

	// Initialize data binding
	chatData = binding.NewStringList()

	// Create menu bar
	createMenuBar() // Sets up the menu bar for myWindow

	// Main UI
	mainUI := makeMainUI()

	// Final layout with menu bar at the top
	mainContainer := container.NewBorder(
		nil,    // Top: Menu bar is managed directly by createMenuBar
		nil,    // Bottom: No specific bottom layout
		nil,    // Left: No specific left layout; handled by mainContent
		nil,    // Right: No specific right layout
		mainUI, // Center: Main UI created by makeMainUI
	)

	myWindow.SetContent(mainContainer)
	myWindow.CenterOnScreen()
	myWindow.Resize(fyne.NewSize(800, 600))
	myWindow.ShowAndRun()
}

// createMenuBar creates the top menu with items Models, Tools, and Settings
func createMenuBar() {
	themeToggle := fyne.NewMenuItem("Toggle Theme", func() {
		pref := myApp.Preferences()
		isDark := !pref.Bool("dark_mode")
		pref.SetBool("dark_mode", isDark)
		setTheme(isDark)
	})

	menu := fyne.NewMainMenu(
		fyne.NewMenu("Models",
			fyne.NewMenuItem("Load Model", func() {
				dialog.ShowInformation("Load Model", "Feature to load a model will be added.", myWindow)
			}),
			fyne.NewMenuItem("Switch Model", func() {
				dialog.ShowInformation("Switch Model", "Feature to switch models will be added.", myWindow)
			}),
		),
		fyne.NewMenu("Tools",
			fyne.NewMenuItem("Token Counter", func() {
				dialog.ShowInformation("Token Counter", "Token counter tool coming soon.", myWindow)
			}),
			fyne.NewMenuItem("Export Chat", func() {
				dialog.ShowInformation("Export Chat", "Export functionality will be added.", myWindow)
			}),
		),
		fyne.NewMenu("Settings",
			fyne.NewMenuItem("Preferences", func() {
				dialog.ShowInformation("Preferences", "Settings menu under construction.", myWindow)
			}),
			fyne.NewMenuItem("About", func() {
				dialog.ShowInformation("About", "Ollama Chat App Version 1.0", myWindow)
			}),
		),
		fyne.NewMenu("Theme", themeToggle),
	)

	myWindow.SetMainMenu(menu)
}

// createChatBubble creates a styled chat bubble
func createChatBubble(message string, isUser bool) *fyne.Container {
	label := widget.NewLabel(message)
	label.Wrapping = fyne.TextWrapWord

	bubble := container.NewStack(
		canvasWithBackgroundAndCenteredInput(label, isUser),
	)

	if isUser {
		return container.NewHBox(layout.NewSpacer(), bubble)
	} else {
		return container.NewHBox(bubble, layout.NewSpacer())
	}
}

// canvasWithBackgroundAndCenteredInput creates a styled background
func canvasWithBackgroundAndCenteredInput(content fyne.CanvasObject, isUser bool) fyne.CanvasObject {
	var bgColor color.Color
	if isUser {
		bgColor = color.RGBA{R: 70, G: 130, B: 180, A: 255}
	} else {
		bgColor = color.Transparent
	}

	background := canvas.NewRectangle(bgColor)
	background.SetMinSize(fyne.NewSize(600, content.MinSize().Height+20))

	return centeredContainer(container.NewStack(background, content))
}

func centeredContainer(content fyne.CanvasObject) fyne.CanvasObject {
	return container.NewVBox(
		layout.NewSpacer(),
		container.New(layout.NewCenterLayout(), container.NewStack(content)),
		layout.NewSpacer(),
	)
}

// createInputComponents creates the message input field and send button
func createInputComponents() (*widget.Entry, *widget.Button) {
	messageInput := widget.NewEntry()
	messageInput.SetPlaceHolder("Type your message here...")

	sendMessage := func() {
		userMessage := strings.TrimSpace(messageInput.Text)
		if len(userMessage) > 500 {
			updateChatData("assistant: Error: Message too long. Please limit to 500 characters.")
			return
		}
		if userMessage != "" {
			// Storing a user message
			updateChatData("user: " + userMessage)
			messageInput.SetText("")
			go handleUserMessage(userMessage)
		}
	}

	messageInput.OnSubmitted = func(content string) {
		sendMessage()
	}

	return messageInput, widget.NewButton("Send", sendMessage)
}

// createChatHistory sets up the scrollable chat history
func createChatHistory() *fyne.Container {
	chatContent := container.NewVBox()
	scroll = container.NewVScroll(chatContent)
	scroll.SetMinSize(fyne.NewSize(400, 600))

	chatData.AddListener(binding.NewDataListener(func() {
		chatContent.Objects = nil
		items, _ := chatData.Get()
		for _, line := range items {
			role, content := parseRoleAndContent(line)
			switch role {
			case "user":
				chatContent.Add(createChatBubble(content, true))
			case "assistant":
				chatContent.Add(createChatBubble(content, false))
				// other roles like "system" can be handled differently if desired
			}
		}
		chatContent.Refresh()
	}))

	return container.New(layout.NewStackLayout(), scroll)
}

// handleUserMessage processes the user's message
func handleUserMessage(userMessage string) {
	if err := streamFromOllama(userMessage); err != nil {
		updateChatData("assistant: Error: " + fmt.Sprintf("%v", err))
	}
}

// updateChatData safely updates the chat data binding
func updateChatData(message string) {
	items, _ := chatData.Get()
	chatData.Set(append(items, message))
}

// streamFromOllama streams the assistant's response
func streamFromOllama(userMessage string) error {
	// Initialize the Ollama API client
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return fmt.Errorf("failed to create Ollama API client: %w", err)
	}

	// Prepare chat messages, including the user's latest message
	messages := []api.Message{
		{Role: "user", Content: userMessage},
	}

	ctx := context.Background()

	// Prepare the chat request
	req := &api.ChatRequest{
		Model:    modelName, // Use your specified model name
		Messages: messages,
	}

	// We'll accumulate the assistant's tokens here until done
	var assistantMessageBuilder strings.Builder
	var assistantIndex int = -1 // We'll store the index of the assistant message line we are updating

	respFunc := func(resp api.ChatResponse) error {
		assistantMessageBuilder.WriteString(resp.Message.Content)

		items, _ := chatData.Get()

		if assistantIndex == -1 {
			// First token for this response, append a new assistant line
			newLine := "assistant: " + assistantMessageBuilder.String()
			items = append(items, newLine)
			chatData.Set(items)
			assistantIndex = len(items) - 1
		} else {
			// Update the existing assistant message line with the current accumulated content
			updatedLine := "assistant: " + assistantMessageBuilder.String()
			items[assistantIndex] = updatedLine
			chatData.Set(items)
			rebuildChatHistory()
		}

		if resp.Done {
			assistantMessageBuilder.Reset()
			assistantIndex = -1
		}
		return nil
	}

	if err := client.Chat(ctx, req, respFunc); err != nil {
		log.Fatal(err)
	}
	return nil
}
