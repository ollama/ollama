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
	mu.Lock()
	defer mu.Unlock()
	existingItems, _ := chatData.Get()
	newItems := []string{
		fmt.Sprintf("Chat %d loaded.", currentChatID+1),
		"Assistant: How can I help you?",
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
	rebuildChatHistory()
}

func rebuildChatHistory() {
	chatContent := scroll.Content.(*fyne.Container)

	mu.Lock()
	defer mu.Unlock()

	chatContent.Objects = nil

	items, _ := chatData.Get()
	for _, message := range items {
		if strings.HasPrefix(message, "You:") {
			chatContent.Add(createChatBubble(message[4:], true))
		} else if strings.HasPrefix(message, "Assistant:") {
			chatContent.Add(createChatBubble(message[10:], false))
		}
	}

	chatContent.Refresh()
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
			updateChatData("Assistant: " + "Error: Message too long. Please limit to 500 characters.")
			return
		}
		if userMessage != "" {
			updateChatData("You: " + userMessage)
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
		mu.Lock()
		defer mu.Unlock()

		chatContent.Objects = nil
		items, _ := chatData.Get()
		for _, message := range items {
			if strings.HasPrefix(message, "You:") {
				chatContent.Add(createChatBubble(message[4:], true))
			} else if strings.HasPrefix(message, "Assistant:") {
				chatContent.Add(createChatBubble(message[10:], false))
			}
		}
		chatContent.Refresh()
	}))

	return container.New(layout.NewStackLayout(), scroll)
}

// handleUserMessage processes the user's message
func handleUserMessage(userMessage string) {
	if err := streamFromOllama(userMessage); err != nil {
		updateChatData("Assistant: " + fmt.Sprintf("Error: %v", err))
	}
}

// updateChatData safely updates the chat data binding
func updateChatData(message string) {

	mu.Lock()
	defer mu.Unlock()
	items, _ := chatData.Get()
	chatData.Set(append(items, message))
	// fmt.Print(items)
	// fmt.Print(message)
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

	// Callback function to handle streaming responses
	respFunc := func(resp api.ChatResponse) error {
		// Print raw response for debugging if desired
		// fmt.Print(resp)

		// Append the streamed content to the builder
		assistantMessageBuilder.WriteString(resp.Message.Content)

		// Once the response is done, we have the full assistant message
		if resp.Done {
			// Now update the chat data with the complete assistant message
			updateChatData("Assistant: " + assistantMessageBuilder.String())
			// Reset the builder for the next response
			assistantMessageBuilder.Reset()
		}

		return nil
	}

	err = client.Chat(ctx, req, respFunc)
	if err != nil {
		log.Fatal(err)
	}
	return nil
}
