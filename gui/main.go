package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"image/color"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

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
	ollamaEndpoint = "http://localhost:11434/api/chat"
	modelName      = "llama3.1"
	httpTimeout    = 30 * time.Second
	appIconPath    = "../app/assets/app.ico"
)

// ChatRequest defines the structure of a request to the Ollama API
type ChatRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
}

// ChatMessage represents a single chat message
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

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
			addAssistantMessage("Error: Message too long. Please limit to 500 characters.")
			return
		}
		if userMessage != "" {
			addUserMessage(userMessage)
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
		addAssistantMessage(fmt.Sprintf("Error: %v", err))
	}
}

// addUserMessage appends a user's message
func addUserMessage(message string) {
	updateChatData("You: " + message)
}

// addAssistantMessage appends an assistant's message
func addAssistantMessage(message string) {
	updateChatData("Assistant: " + message)
}

// updateChatData safely updates the chat data binding
func updateChatData(message string) {
	mu.Lock()
	defer mu.Unlock()
	items, _ := chatData.Get()
	chatData.Set(append(items, message))
}

// streamFromOllama streams the assistant's response
func streamFromOllama(userMessage string) error {
	requestBody := ChatRequest{
		Model:    modelName,
		Messages: []ChatMessage{{Role: "user", Content: userMessage}},
	}

	body, err := json.Marshal(requestBody)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	client := &http.Client{Timeout: httpTimeout}
	req, err := http.NewRequest("POST", ollamaEndpoint, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API error: %s", string(b))
	}

	return processSSEStream(resp.Body)
}

// processSSEStream processes the server-sent events stream
func processSSEStream(body io.ReadCloser) error {
	scanner := bufio.NewScanner(body)
	var partialResponse, finalDetails string

	for scanner.Scan() {
		line := scanner.Text()
		fmt.Printf("SSE Line: %s\n", line)

		var chatResponse struct {
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			Done               bool   `json:"done"`
			DoneReason         string `json:"done_reason,omitempty"`
			TotalDuration      int64  `json:"total_duration,omitempty"`
			LoadDuration       int64  `json:"load_duration,omitempty"`
			PromptEvalCount    int    `json:"prompt_eval_count,omitempty"`
			PromptEvalDuration int64  `json:"prompt_eval_duration,omitempty"`
			EvalCount          int    `json:"eval_count,omitempty"`
			EvalDuration       int64  `json:"eval_duration,omitempty"`
		}

		if err := json.Unmarshal([]byte(line), &chatResponse); err != nil {
			fmt.Printf("Failed to unmarshal JSON: %v\n", err)
			continue
		}

		partialResponse += chatResponse.Message.Content

		if chatResponse.Done {
			finalDetails = fmt.Sprintf(
				"Done Reason: %s\nTotal Duration: %d ms\nLoad Duration: %d ms\nPrompt Eval Count: %d\nPrompt Eval Duration: %d ms\nEval Count: %d\nEval Duration: %d ms",
				chatResponse.DoneReason,
				chatResponse.TotalDuration,
				chatResponse.LoadDuration,
				chatResponse.PromptEvalCount,
				chatResponse.PromptEvalDuration,
				chatResponse.EvalCount,
				chatResponse.EvalDuration,
			)
		}

		// Scroll only if the user is at the bottom
		if scroll.Offset.Y+scroll.Size().Height >= scroll.Content.Size().Height {
			scroll.ScrollToBottom()
		}
	}

	if partialResponse != "" {
		updateChatData(fmt.Sprintf("Assistant: %s\n\nModel: %s \n\n%s", partialResponse, modelName, finalDetails))
	}

	if scanner.Err() != nil {
		return fmt.Errorf("error reading stream: %w", scanner.Err())
	}

	return nil
}
