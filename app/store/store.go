//go:build windows || darwin

// Package store provides a simple JSON file store for the desktop application
// to save and load data such as ollama server configuration, messages,
// login information and more.
package store

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/ollama/ollama/app/types/not"
)

type File struct {
	Filename string `json:"filename"`
	Data     []byte `json:"data"`
}

type User struct {
	Name     string    `json:"name"`
	Email    string    `json:"email"`
	Plan     string    `json:"plan"`
	CachedAt time.Time `json:"cachedAt"`
}

type Message struct {
	Role              string           `json:"role"`
	Content           string           `json:"content"`
	Thinking          string           `json:"thinking"`
	Stream            bool             `json:"stream"`
	Model             string           `json:"model,omitempty"`
	Attachments       []File           `json:"attachments,omitempty"`
	ToolCalls         []ToolCall       `json:"tool_calls,omitempty"`
	ToolCall          *ToolCall        `json:"tool_call,omitempty"`
	ToolName          string           `json:"tool_name,omitempty"`
	ToolResult        *json.RawMessage `json:"tool_result,omitempty"`
	CreatedAt         time.Time        `json:"created_at"`
	UpdatedAt         time.Time        `json:"updated_at"`
	ThinkingTimeStart *time.Time       `json:"thinkingTimeStart,omitempty" ts_type:"Date | undefined" ts_transform:"__VALUE__ && new Date(__VALUE__)"`
	ThinkingTimeEnd   *time.Time       `json:"thinkingTimeEnd,omitempty" ts_type:"Date | undefined" ts_transform:"__VALUE__ && new Date(__VALUE__)"`
}

// MessageOptions contains optional parameters for creating a Message
type MessageOptions struct {
	Model             string
	Attachments       []File
	Stream            bool
	Thinking          string
	ToolCalls         []ToolCall
	ToolCall          *ToolCall
	ToolResult        *json.RawMessage
	ThinkingTimeStart *time.Time
	ThinkingTimeEnd   *time.Time
}

// NewMessage creates a new Message with the given options
func NewMessage(role, content string, opts *MessageOptions) Message {
	now := time.Now()
	msg := Message{
		Role:      role,
		Content:   content,
		CreatedAt: now,
		UpdatedAt: now,
	}

	if opts != nil {
		msg.Model = opts.Model
		msg.Attachments = opts.Attachments
		msg.Stream = opts.Stream
		msg.Thinking = opts.Thinking
		msg.ToolCalls = opts.ToolCalls
		msg.ToolCall = opts.ToolCall
		msg.ToolResult = opts.ToolResult
		msg.ThinkingTimeStart = opts.ThinkingTimeStart
		msg.ThinkingTimeEnd = opts.ThinkingTimeEnd
	}

	return msg
}

type ToolCall struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Result    any    `json:"result,omitempty"`
}

type Model struct {
	Model      string     `json:"model"`                 // Model name
	Digest     string     `json:"digest,omitempty"`      // Model digest from the registry
	ModifiedAt *time.Time `json:"modified_at,omitempty"` // When the model was last modified locally
}

type Chat struct {
	ID           string          `json:"id"`
	Messages     []Message       `json:"messages"`
	Title        string          `json:"title"`
	CreatedAt    time.Time       `json:"created_at"`
	BrowserState json.RawMessage `json:"browser_state,omitempty" ts_type:"BrowserStateData"`
}

// NewChat creates a new Chat with the ID, with CreatedAt timestamp initialized
func NewChat(id string) *Chat {
	return &Chat{
		ID:        id,
		Messages:  []Message{},
		CreatedAt: time.Now(),
	}
}

type Settings struct {
	// Expose is a boolean that indicates if the ollama server should
	// be exposed to the network
	Expose bool

	// Browser is a boolean that indicates if the ollama server should
	// be exposed to browser windows (e.g. CORS set to allow all origins)
	Browser bool

	// Survey is a boolean that indicates if the user allows anonymous
	// inference information to be shared with Ollama
	Survey bool

	// Models is a string that contains the models to load on startup
	Models string

	// TODO(parthsareen): temporary for experimentation
	// Agent indicates if the app should use multi-turn tools to fulfill user requests
	Agent bool

	// Tools indicates if the app should use single-turn tools to fulfill user requests
	Tools bool

	// WorkingDir specifies the working directory for all agent operations
	WorkingDir string

	// ContextLength specifies the context length for the ollama server (using OLLAMA_CONTEXT_LENGTH)
	ContextLength int

	// AirplaneMode when true, turns off Ollama Turbo features and only uses local models
	AirplaneMode bool

	// TurboEnabled indicates if Ollama Turbo features are enabled
	TurboEnabled bool

	// Maps gpt-oss specific frontend name' BrowserToolEnabled' to db field 'websearch_enabled'
	WebSearchEnabled bool

	// ThinkEnabled indicates if thinking is enabled
	ThinkEnabled bool

	// ThinkLevel indicates the level of thinking to use for models that support multiple levels
	ThinkLevel string

	// SelectedModel stores the last model that the user selected
	SelectedModel string

	// SidebarOpen indicates if the chat sidebar is open
	SidebarOpen bool
}

type Store struct {
	// DBPath allows overriding the default database path (mainly for testing)
	DBPath string

	// dbMu protects database initialization only
	dbMu sync.Mutex
	db   *database
}

var defaultDBPath = func() string {
	switch runtime.GOOS {
	case "windows":
		return filepath.Join(os.Getenv("LOCALAPPDATA"), "Ollama", "db.sqlite")
	case "darwin":
		return filepath.Join(os.Getenv("HOME"), "Library", "Application Support", "Ollama", "db.sqlite")
	default:
		return filepath.Join(os.Getenv("HOME"), ".ollama", "db.sqlite")
	}
}()

// legacyConfigPath is the path to the old config.json file
var legacyConfigPath = func() string {
	switch runtime.GOOS {
	case "windows":
		return filepath.Join(os.Getenv("LOCALAPPDATA"), "Ollama", "config.json")
	case "darwin":
		return filepath.Join(os.Getenv("HOME"), "Library", "Application Support", "Ollama", "config.json")
	default:
		return filepath.Join(os.Getenv("HOME"), ".ollama", "config.json")
	}
}()

// legacyData represents the old config.json structure (only fields we need to migrate)
type legacyData struct {
	ID           string `json:"id"`
	FirstTimeRun bool   `json:"first-time-run"`
}

func (s *Store) ensureDB() error {
	// Fast path: check if db is already initialized
	if s.db != nil {
		return nil
	}

	// Slow path: initialize database with lock
	s.dbMu.Lock()
	defer s.dbMu.Unlock()

	// Double-check after acquiring lock
	if s.db != nil {
		return nil
	}

	dbPath := s.DBPath
	if dbPath == "" {
		dbPath = defaultDBPath
	}

	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(dbPath), 0o755); err != nil {
		return fmt.Errorf("create db directory: %w", err)
	}

	database, err := newDatabase(dbPath)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}

	// Generate device ID if needed
	id, err := database.getID()
	if err != nil || id == "" {
		// Generate new UUID for device
		u, err := uuid.NewV7()
		if err == nil {
			database.setID(u.String())
		}
	}

	s.db = database

	// Check if we need to migrate from config.json
	migrated, err := database.isConfigMigrated()
	if err != nil || !migrated {
		if err := s.migrateFromConfig(database); err != nil {
			slog.Warn("failed to migrate from config.json", "error", err)
		}
	}

	return nil
}

// migrateFromConfig attempts to migrate ID and FirstTimeRun from config.json
func (s *Store) migrateFromConfig(database *database) error {
	configPath := legacyConfigPath

	// Check if config.json exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		// No config to migrate, mark as migrated
		return database.setConfigMigrated(true)
	}

	// Read the config file
	b, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("read legacy config: %w", err)
	}

	var legacy legacyData
	if err := json.Unmarshal(b, &legacy); err != nil {
		// If we can't parse it, just mark as migrated and move on
		slog.Warn("failed to parse legacy config.json", "error", err)
		return database.setConfigMigrated(true)
	}

	// Migrate the ID if present
	if legacy.ID != "" {
		if err := database.setID(legacy.ID); err != nil {
			return fmt.Errorf("migrate device ID: %w", err)
		}
		slog.Info("migrated device ID from config.json")
	}

	hasCompleted := legacy.FirstTimeRun // If old FirstTimeRun is true, it means first run was completed
	if err := database.setHasCompletedFirstRun(hasCompleted); err != nil {
		return fmt.Errorf("migrate first time run: %w", err)
	}
	slog.Info("migrated first run status from config.json", "hasCompleted", hasCompleted)

	// Mark as migrated
	if err := database.setConfigMigrated(true); err != nil {
		return fmt.Errorf("mark config as migrated: %w", err)
	}

	slog.Info("successfully migrated settings from config.json")
	return nil
}

func (s *Store) ID() (string, error) {
	if err := s.ensureDB(); err != nil {
		return "", err
	}

	return s.db.getID()
}

func (s *Store) HasCompletedFirstRun() (bool, error) {
	if err := s.ensureDB(); err != nil {
		return false, err
	}

	return s.db.getHasCompletedFirstRun()
}

func (s *Store) SetHasCompletedFirstRun(hasCompleted bool) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	return s.db.setHasCompletedFirstRun(hasCompleted)
}

func (s *Store) Settings() (Settings, error) {
	if err := s.ensureDB(); err != nil {
		return Settings{}, fmt.Errorf("load settings: %w", err)
	}

	settings, err := s.db.getSettings()
	if err != nil {
		return Settings{}, err
	}

	// Set default models directory if not set
	if settings.Models == "" {
		dir := os.Getenv("OLLAMA_MODELS")
		if dir != "" {
			settings.Models = dir
		} else {
			home, err := os.UserHomeDir()
			if err == nil {
				settings.Models = filepath.Join(home, ".ollama", "models")
			}
		}
	}

	return settings, nil
}

func (s *Store) SetSettings(settings Settings) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	return s.db.setSettings(settings)
}

func (s *Store) Chats() ([]Chat, error) {
	if err := s.ensureDB(); err != nil {
		return nil, err
	}

	return s.db.getAllChats()
}

func (s *Store) Chat(id string) (*Chat, error) {
	return s.ChatWithOptions(id, true)
}

func (s *Store) ChatWithOptions(id string, loadAttachmentData bool) (*Chat, error) {
	if err := s.ensureDB(); err != nil {
		return nil, err
	}

	chat, err := s.db.getChatWithOptions(id, loadAttachmentData)
	if err != nil {
		return nil, fmt.Errorf("%w: chat %s", not.Found, id)
	}

	return chat, nil
}

func (s *Store) SetChat(chat Chat) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	return s.db.saveChat(chat)
}

func (s *Store) DeleteChat(id string) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	// Delete from database
	if err := s.db.deleteChat(id); err != nil {
		return fmt.Errorf("%w: chat %s", not.Found, id)
	}

	// Also delete associated images
	chatImgDir := filepath.Join(s.ImgDir(), id)
	if err := os.RemoveAll(chatImgDir); err != nil {
		// Log error but don't fail the deletion
		slog.Warn("failed to delete chat images", "chat_id", id, "error", err)
	}

	return nil
}

func (s *Store) WindowSize() (int, int, error) {
	if err := s.ensureDB(); err != nil {
		return 0, 0, err
	}

	return s.db.getWindowSize()
}

func (s *Store) SetWindowSize(width, height int) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	return s.db.setWindowSize(width, height)
}

func (s *Store) UpdateLastMessage(chatID string, message Message) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	return s.db.updateLastMessage(chatID, message)
}

func (s *Store) AppendMessage(chatID string, message Message) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	return s.db.appendMessage(chatID, message)
}

func (s *Store) UpdateChatBrowserState(chatID string, state json.RawMessage) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	return s.db.updateChatBrowserState(chatID, state)
}

func (s *Store) User() (*User, error) {
	if err := s.ensureDB(); err != nil {
		return nil, err
	}

	return s.db.getUser()
}

func (s *Store) SetUser(user User) error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	user.CachedAt = time.Now()
	return s.db.setUser(user)
}

func (s *Store) ClearUser() error {
	if err := s.ensureDB(); err != nil {
		return err
	}

	return s.db.clearUser()
}

func (s *Store) Close() error {
	s.dbMu.Lock()
	defer s.dbMu.Unlock()

	if s.db != nil {
		return s.db.Close()
	}
	return nil
}
