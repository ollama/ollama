package main

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"fmt"
	"image/color"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3" // SQLite driver
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

const (
	modelName   = "llama3.1"
	httpTimeout = 30 * time.Second
	appIconPath = "../app/assets/app.ico"
	dbFile      = "ollama_chats.db"
)

var (
	myApp         fyne.App
	myWindow      fyne.Window
	chatData      binding.StringList
	scroll        *container.Scroll
	currentChatID int
	db            *sql.DB
	mu            sync.Mutex
)

func main() {
	initializeDB()
	defer db.Close()

	initializeApp()
}

func initializeDB() {
	var err error
	db, err = sql.Open("sqlite3", dbFile)
	if err != nil {
		log.Fatalf("Failed to open database: %v", err)
	}

	// Create tables if they do not exist
	createChatsTable := `
	CREATE TABLE IF NOT EXISTS chats (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		title TEXT,
		hash TEXT
	);
	`

	createMessagesTable := `
	CREATE TABLE IF NOT EXISTS messages (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		chat_id INTEGER,
		role TEXT,
		content TEXT,
		FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
	);
	`

	_, err = db.Exec(createChatsTable)
	if err != nil {
		log.Fatalf("Failed to create chats table: %v", err)
	}

	_, err = db.Exec(createMessagesTable)
	if err != nil {
		log.Fatalf("Failed to create messages table: %v", err)
	}
}

func saveCurrentChat() {
	mu.Lock()
	defer mu.Unlock()

	if currentChatID < 0 {
		return
	}

	currentChat := getCurrentChat()
	if len(currentChat) == 0 {
		return
	}
	hash := hashChat(currentChat)

	// Check if chat already exists in DB
	var existingHash string
	err := db.QueryRow("SELECT hash FROM chats WHERE id = ?", currentChatID).Scan(&existingHash)

	if err == sql.ErrNoRows {
		// Insert new chat
		title := fmt.Sprintf("Chat %d", getNextChatNumber())
		res, err := db.Exec("INSERT INTO chats (title, hash) VALUES (?, ?)", title, hash)
		if err != nil {
			log.Printf("Failed to insert new chat: %v", err)
			return
		}
		newID, _ := res.LastInsertId()
		currentChatID = int(newID)
	} else if err != nil {
		log.Printf("Error checking chat: %v", err)
		return
	} else {
		// Existing chat, update hash if needed
		if existingHash != hash {
			_, err := db.Exec("UPDATE chats SET hash = ? WHERE id = ?", hash, currentChatID)
			if err != nil {
				log.Printf("Failed to update chat hash: %v", err)
			}
		}
	}

	// Delete old messages for this chat
	_, err = db.Exec("DELETE FROM messages WHERE chat_id = ?", currentChatID)
	if err != nil {
		log.Printf("Failed to delete old messages: %v", err)
	}

	// Insert updated messages
	for _, line := range currentChat {
		role, content := parseRoleAndContent(line)
		_, err := db.Exec("INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)", currentChatID, role, content)
		if err != nil {
			log.Printf("Failed to insert message: %v", err)
		}
	}

	updateSidebar()
}

func hashChat(chat []string) string {
	hash := sha256.New()
	for _, line := range chat {
		hash.Write([]byte(line))
	}
	return fmt.Sprintf("%x", hash.Sum(nil))
}

func getCurrentChat() []string {
	items, _ := chatData.Get()
	return items
}

func loadChatHistory(chatID int) {
	mu.Lock()
	defer mu.Unlock()

	rows, err := db.Query("SELECT role, content FROM messages WHERE chat_id = ?", chatID)
	if err != nil {
		log.Printf("Failed to load chat messages: %v", err)
		return
	}
	defer rows.Close()

	var messages []string
	for rows.Next() {
		var role, content string
		if err := rows.Scan(&role, &content); err != nil {
			log.Printf("Error scanning message: %v", err)
			continue
		}
		messages = append(messages, role+": "+content)
	}
	chatData.Set(messages)
}

// updateSidebar refreshes the chat list in the sidebar
func updateSidebar() {
	chatsList, err := loadChatList()
	if err != nil {
		log.Printf("Failed to load chat list: %v", err)
		return
	}

	serverList := widget.NewList(
		func() int { return len(chatsList) + 1 },
		func() fyne.CanvasObject {
			label := widget.NewLabel("")
			deleteButton := widget.NewButtonWithIcon("", theme.DeleteIcon(), nil)
			hbox := container.NewBorder(nil, nil, nil, deleteButton, label)
			return hbox
		},
		func(id widget.ListItemID, o fyne.CanvasObject) {
			border := o.(*fyne.Container)
			var label *widget.Label
			var deleteBtn *widget.Button
			for _, obj := range border.Objects {
				switch c := obj.(type) {
				case *widget.Label:
					label = c
				case *widget.Button:
					deleteBtn = c
				}
			}
			if id == 0 {
				label.SetText("New Chat")
				deleteBtn.Hide()
				deleteBtn.OnTapped = nil
			} else {
				chat := chatsList[id-1]
				label.SetText(chat.title)
				deleteBtn.Show()
				deleteBtn.OnTapped = func() {
					deleteChat(chat.id)
				}
			}
		},
	)

	serverList.OnSelected = func(id widget.ListItemID) {
		if id == 0 {
			handleNewChatClick()
		} else {
			chat := chatsList[id-1]
			handleSavedChatClick(chat.id)
		}
	}

	mainContent := makeMainUI(serverList)
	myWindow.SetContent(mainContent)
}

type chatRecord struct {
	id    int
	title string
	hash  string
}

func loadChatList() ([]chatRecord, error) {
	rows, err := db.Query("SELECT id, title, hash FROM chats ORDER BY id")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var result []chatRecord
	for rows.Next() {
		var r chatRecord
		if err := rows.Scan(&r.id, &r.title, &r.hash); err != nil {
			log.Printf("Error scanning chat list: %v", err)
			continue
		}
		result = append(result, r)
	}
	return result, nil
}

func getNextChatNumber() int {
	// Count how many chats are in DB
	var count int
	err := db.QueryRow("SELECT COUNT(*) FROM chats").Scan(&count)
	if err != nil {
		log.Printf("Failed to count chats: %v", err)
	}
	return count + 1
}

func handleNewChatClick() {
	saveCurrentChat()

	// Create a new chat
	newTitle := fmt.Sprintf("Chat %d", getNextChatNumber())
	res, err := db.Exec("INSERT INTO chats (title, hash) VALUES (?, ?)", newTitle, "")
	if err != nil {
		log.Printf("Failed to create new chat: %v", err)
		return
	}

	newID, _ := res.LastInsertId()
	currentChatID = int(newID)
	initialMessage := []string{"assistant: Welcome to your new chat!"}
	chatData.Set(initialMessage)
	saveCurrentChat()
	updateSidebar()
}

func handleSavedChatClick(chatID int) {
	saveCurrentChat()
	currentChatID = chatID
	loadChatHistory(chatID)
}

// Delete a chat from the database
func deleteChat(chatID int) {
	_, err := db.Exec("DELETE FROM chats WHERE id = ?", chatID)
	if err != nil {
		log.Printf("Failed to delete chat: %v", err)
	}

	// If we deleted the current chat, reset to a default state
	if chatID == currentChatID {
		// Load another chat if available
		chatsList, _ := loadChatList()
		if len(chatsList) > 0 {
			currentChatID = chatsList[0].id
			loadChatHistory(currentChatID)
		} else {
			// No chats left
			currentChatID = -1
			chatData.Set([]string{})
		}
	}

	updateSidebar()
}

// makeMainUI creates the primary UI layout
func makeMainUI(serverList *widget.List) fyne.CanvasObject {
	chatHistory := createChatHistory()
	messageInput, sendButton := createInputComponents()

	uploadButton := widget.NewButton("+", func() {
		dialog.ShowInformation("File Upload", "Feature to upload files will be added.", myWindow)
	})

	inputContainer := container.NewBorder(nil, nil, uploadButton, sendButton, messageInput)
	messagePane := container.NewBorder(nil, inputContainer, nil, nil, chatHistory)
	mainContent := container.NewHSplit(serverList, messagePane)
	mainContent.SetOffset(0.2)

	return mainContent
}

// setTheme toggles between light and dark themes
func setTheme(isDark bool) {
	if isDark {
		myApp.Settings().SetTheme(theme.DarkTheme())
	} else {
		myApp.Settings().SetTheme(theme.LightTheme())
	}
}

func rebuildChatHistory() {
	chatContent := scroll.Content.(*fyne.Container)

	chatContent.Objects = nil

	items, _ := chatData.Get()
	for _, message := range items {
		role, content := parseRoleAndContent(message)
		isUser := (role == "user")

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

	appIcon := loadAppIcon(appIconPath)
	if appIcon != nil {
		myWindow.SetIcon(appIcon)
	}
	createMenuBar()

	chatData = binding.NewStringList()

	// If no chats exist, create a default one
	var chatCount int
	err := db.QueryRow("SELECT COUNT(*) FROM chats").Scan(&chatCount)
	if err != nil {
		log.Printf("Failed to count chats: %v", err)
	}
	if chatCount == 0 {
		// Insert initial chat
		res, err := db.Exec("INSERT INTO chats (title, hash) VALUES (?, ?)", "Welcome Chat", "")
		if err != nil {
			log.Printf("Failed to create initial chat: %v", err)
		}
		newID, _ := res.LastInsertId()
		currentChatID = int(newID)
		initialMessage := []string{"assistant: Welcome to the chat!"}
		chatData.Set(initialMessage)
		saveCurrentChat()
	} else {
		// Load the first chat by default
		var firstChatID int
		err := db.QueryRow("SELECT id FROM chats ORDER BY id LIMIT 1").Scan(&firstChatID)
		if err == nil {
			currentChatID = firstChatID
			loadChatHistory(firstChatID)
		}
	}

	chatsList, _ := loadChatList()

	serverList := widget.NewList(
		func() int { return len(chatsList) + 1 },
		func() fyne.CanvasObject {
			label := widget.NewLabel("")
			deleteButton := widget.NewButtonWithIcon("", theme.DeleteIcon(), nil)
			hbox := container.NewBorder(nil, nil, nil, deleteButton, label)
			return hbox
		},
		func(id widget.ListItemID, o fyne.CanvasObject) {
			border := o.(*fyne.Container)
			var label *widget.Label
			var deleteBtn *widget.Button
			for _, obj := range border.Objects {
				switch c := obj.(type) {
				case *widget.Label:
					label = c
				case *widget.Button:
					deleteBtn = c
				}
			}
			if id == 0 {
				label.SetText("New Chat")
				deleteBtn.Hide()
				deleteBtn.OnTapped = nil
			} else {
				chat := chatsList[id-1]
				label.SetText(chat.title)
				deleteBtn.Show()
				deleteBtn.OnTapped = func() {
					deleteChat(chat.id)
				}
			}
		},
	)

	serverList.OnSelected = func(id widget.ListItemID) {
		if id == 0 {
			handleNewChatClick()
		} else {
			chatsList, _ := loadChatList()
			handleSavedChatClick(chatsList[id-1].id)
		}
	}

	mainUI := makeMainUI(serverList)

	myWindow.SetContent(mainUI)
	myWindow.CenterOnScreen()
	myWindow.Resize(fyne.NewSize(800, 600))
	myWindow.ShowAndRun()
}

// createMenuBar creates the top menu
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
	saveCurrentChat()
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
		Model:    modelName,
		Messages: messages,
	}

	var assistantMessageBuilder strings.Builder
	var assistantIndex int = -1

	respFunc := func(resp api.ChatResponse) error {
		assistantMessageBuilder.WriteString(resp.Message.Content)

		items, _ := chatData.Get()

		if assistantIndex == -1 {
			newLine := "assistant: " + assistantMessageBuilder.String()
			items = append(items, newLine)
			chatData.Set(items)
			assistantIndex = len(items) - 1
		} else {
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
