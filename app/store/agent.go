package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

type AgentChat struct {
	ID        string
	Title     string
	Model     string
	CreatedAt time.Time
	Messages  []api.Message
}

type ChatSummary struct {
	ID           string
	Title        string
	Model        string
	CreatedAt    time.Time
	UpdatedAt    time.Time
	MessageCount int
	ApproxBytes  int64
}

func New(path string) (*Store, error) {
	store := &Store{DBPath: path}
	if err := store.ensureDB(); err != nil {
		return nil, err
	}
	return store, nil
}

func (s *Store) EnsureChat(ctx context.Context, id string, title string) error {
	if id == "" {
		return fmt.Errorf("chat id is required")
	}
	if err := s.ensureDB(); err != nil {
		return err
	}
	_, err := s.db.conn.ExecContext(ctx, `
		INSERT INTO chats (id, title, created_at)
		VALUES (?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			title = CASE
				WHEN excluded.title != '' THEN excluded.title
				ELSE chats.title
			END
	`, id, title, time.Now())
	if err != nil {
		return fmt.Errorf("ensure chat: %w", err)
	}
	return nil
}

func (s *Store) SetChatModel(ctx context.Context, chatID string, model string) error {
	chatID = strings.TrimSpace(chatID)
	model = strings.TrimSpace(model)
	if chatID == "" {
		return fmt.Errorf("chat id is required")
	}
	if model == "" {
		return fmt.Errorf("model is required")
	}
	if err := s.EnsureChat(ctx, chatID, ""); err != nil {
		return err
	}
	if _, err := s.db.conn.ExecContext(ctx, `UPDATE chats SET model_name = ? WHERE id = ?`, model, chatID); err != nil {
		return fmt.Errorf("set chat model: %w", err)
	}
	return nil
}

func (s *Store) AppendAgentMessage(ctx context.Context, chatID string, msg api.Message, model string) error {
	if err := s.EnsureChat(ctx, chatID, ""); err != nil {
		return err
	}

	tx, err := s.db.conn.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	messageID, err := insertAgentMessage(ctx, tx, chatID, msg, model)
	if err != nil {
		return err
	}
	for _, toolCall := range msg.ToolCalls {
		if err := insertAgentToolCall(ctx, tx, messageID, toolCall); err != nil {
			return err
		}
	}

	if msg.Role == "user" && strings.TrimSpace(msg.Content) != "" {
		if err := maybeSetAgentTitle(ctx, tx, chatID, msg.Content); err != nil {
			return err
		}
	}

	return tx.Commit()
}

func (s *Store) UpdateLastAgentMessage(ctx context.Context, chatID string, msg api.Message, model string) error {
	if err := s.ensureDB(); err != nil {
		return err
	}
	tx, err := s.db.conn.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	var messageID int64
	if err := tx.QueryRowContext(ctx, `SELECT COALESCE(MAX(id), 0) FROM messages WHERE chat_id = ?`, chatID).Scan(&messageID); err != nil {
		return fmt.Errorf("get last message id: %w", err)
	}
	if messageID == 0 {
		return fmt.Errorf("no message found to update")
	}

	now := time.Now()
	modelName := sql.NullString{}
	if model != "" {
		modelName = sql.NullString{String: model, Valid: true}
	}

	imagesJSON, err := marshalAgentMessageImages(msg.Images)
	if err != nil {
		return err
	}

	_, err = tx.ExecContext(ctx, `
		UPDATE messages
		SET role = ?, content = ?, thinking = ?, images = ?, tool_name = ?, tool_call_id = ?, model_name = ?, updated_at = ?
		WHERE id = ?
	`, msg.Role, msg.Content, msg.Thinking, imagesJSON, msg.ToolName, msg.ToolCallID, modelName, now, messageID)
	if err != nil {
		return fmt.Errorf("update last message: %w", err)
	}

	if _, err := tx.ExecContext(ctx, `DELETE FROM tool_calls WHERE message_id = ?`, messageID); err != nil {
		return fmt.Errorf("delete old tool calls: %w", err)
	}
	for _, toolCall := range msg.ToolCalls {
		if err := insertAgentToolCall(ctx, tx, messageID, toolCall); err != nil {
			return err
		}
	}

	return tx.Commit()
}

func (s *Store) AgentChat(ctx context.Context, id string) (*AgentChat, error) {
	if err := s.ensureDB(); err != nil {
		return nil, err
	}
	var chat AgentChat
	var chatModel string
	if err := s.db.conn.QueryRowContext(ctx, `
		SELECT id, title, model_name, created_at FROM chats WHERE id = ?
	`, id).Scan(&chat.ID, &chat.Title, &chatModel, &chat.CreatedAt); err != nil {
		return nil, err
	}
	if strings.TrimSpace(chatModel) != "" {
		chat.Model = chatModel
	} else {
		model, err := latestAgentModelForChat(ctx, s.db.conn, id)
		if err != nil {
			return nil, err
		}
		chat.Model = model
	}

	rows, err := s.db.conn.QueryContext(ctx, `
		SELECT id, role, content, thinking, images, tool_name, tool_call_id FROM messages WHERE chat_id = ? AND archived = 0 ORDER BY id ASC
	`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var messageID int64
		var msg api.Message
		var imagesJSON string
		if err := rows.Scan(&messageID, &msg.Role, &msg.Content, &msg.Thinking, &imagesJSON, &msg.ToolName, &msg.ToolCallID); err != nil {
			return nil, err
		}
		images, err := unmarshalAgentMessageImages(imagesJSON)
		if err != nil {
			return nil, err
		}
		msg.Images = images
		toolCalls, err := getAgentToolCalls(ctx, s.db.conn, messageID)
		if err != nil {
			return nil, err
		}
		msg.ToolCalls = toolCalls
		chat.Messages = append(chat.Messages, msg)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	summary, err := latestCompactionSummary(ctx, s.db.conn, id)
	if err != nil {
		return nil, err
	}
	if summary != "" && !messagesContainCompactionSummary(chat.Messages) {
		chat.Messages = insertCompactionSummaryAfterLeadingSystemMessages(chat.Messages, agent.CompactionSummaryMessages(summary, false))
	} else {
		chat.Messages = moveCompactionSummaryBeforeKeptMessages(chat.Messages)
	}

	return &chat, nil
}

func (s *Store) LatestChat(ctx context.Context) (*AgentChat, error) {
	if err := s.ensureDB(); err != nil {
		return nil, err
	}
	var chatID string
	query := fmt.Sprintf(`
			SELECT c.id
			FROM chats c
			JOIN messages m ON m.chat_id = c.id
			GROUP BY c.id
			HAVING %[1]s IS NOT NULL
			ORDER BY MAX(m.updated_at) DESC, MAX(m.id) DESC
			LIMIT 1
		`, currentAgentModelSelectExpr("c"))
	if err := s.db.conn.QueryRowContext(ctx, query).Scan(&chatID); err != nil {
		return nil, err
	}
	return s.AgentChat(ctx, chatID)
}

func (s *Store) LatestChatForModel(ctx context.Context, model string) (*AgentChat, error) {
	if strings.TrimSpace(model) == "" {
		return nil, fmt.Errorf("model is required")
	}
	if err := s.ensureDB(); err != nil {
		return nil, err
	}

	var chatID string
	query := fmt.Sprintf(`
			SELECT c.id
			FROM chats c
			JOIN messages m ON m.chat_id = c.id
			GROUP BY c.id
			HAVING %[1]s = ?
			ORDER BY MAX(m.updated_at) DESC, MAX(m.id) DESC
			LIMIT 1
		`, currentAgentModelSelectExpr("c"))
	if err := s.db.conn.QueryRowContext(ctx, query, model).Scan(&chatID); err != nil {
		return nil, err
	}
	return s.AgentChat(ctx, chatID)
}

func (s *Store) ListChats(ctx context.Context, limit int) ([]ChatSummary, error) {
	if err := s.ensureDB(); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 50
	}

	query := fmt.Sprintf(`
			SELECT
				c.id,
				c.title,
				c.created_at,
				MAX(m.updated_at) AS updated_at,
				COUNT(m.id) AS message_count,
				COALESCE(SUM(
					LENGTH(m.role) +
					LENGTH(m.content) +
					LENGTH(m.thinking) +
					LENGTH(m.tool_name) +
					LENGTH(m.tool_call_id)
				), 0) AS approx_bytes,
				%[1]s AS current_model
			FROM chats c
			JOIN messages m ON m.chat_id = c.id AND m.archived = 0
			GROUP BY c.id
			ORDER BY updated_at DESC, MAX(m.id) DESC
			LIMIT ?
		`, currentAgentModelSelectExpr("c"))
	rows, err := s.db.conn.QueryContext(ctx, query, limit)
	if err != nil {
		return nil, fmt.Errorf("list chats: %w", err)
	}
	defer rows.Close()

	var summaries []ChatSummary
	for rows.Next() {
		var summary ChatSummary
		var updatedAt string
		var modelName sql.NullString
		if err := rows.Scan(&summary.ID, &summary.Title, &summary.CreatedAt, &updatedAt, &summary.MessageCount, &summary.ApproxBytes, &modelName); err != nil {
			return nil, fmt.Errorf("scan chat summary: %w", err)
		}
		if modelName.Valid {
			summary.Model = modelName.String
		}
		summary.UpdatedAt, err = parseAgentSQLiteTime(updatedAt)
		if err != nil {
			return nil, fmt.Errorf("parse chat updated_at: %w", err)
		}
		summaries = append(summaries, summary)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read chat summaries: %w", err)
	}

	return summaries, nil
}

func (s *Store) ListUserMessages(ctx context.Context, limit int) ([]string, error) {
	if err := s.ensureDB(); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 50
	}

	rows, err := s.db.conn.QueryContext(ctx, `
		SELECT content
		FROM (
			SELECT id, content
			FROM messages
			WHERE role = 'user'
				AND TRIM(content) != ''
				AND content NOT LIKE ?
			ORDER BY id DESC
			LIMIT ?
		)
		ORDER BY id ASC
	`, agent.CompactionSummaryMessagePrefix+"%", limit)
	if err != nil {
		return nil, fmt.Errorf("list user messages: %w", err)
	}
	defer rows.Close()

	var messages []string
	for rows.Next() {
		var content string
		if err := rows.Scan(&content); err != nil {
			return nil, fmt.Errorf("scan user message: %w", err)
		}
		messages = append(messages, content)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read user messages: %w", err)
	}
	return messages, nil
}

func (s *Store) ArchiveForCompaction(ctx context.Context, chatID string, keepUserTurns int, summary string, continueTask bool) error {
	return s.archiveForCompaction(ctx, chatID, keepUserTurns, summary, continueTask)
}

func (s *Store) archiveForCompaction(ctx context.Context, chatID string, keepUserTurns int, summary string, continueTask bool) error {
	if err := s.ensureDB(); err != nil {
		return err
	}
	if chatID == "" {
		return fmt.Errorf("chat id is required")
	}
	if keepUserTurns < 0 {
		return fmt.Errorf("keep user turns must be non-negative")
	}
	if strings.TrimSpace(summary) == "" {
		return fmt.Errorf("summary is required")
	}

	tx, err := s.db.conn.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	var keepStartID int64
	if keepUserTurns == 0 {
		if err := tx.QueryRowContext(ctx, `
			SELECT COALESCE(MAX(id) + 1, 0)
			FROM messages
			WHERE chat_id = ? AND archived = 0
		`, chatID).Scan(&keepStartID); err != nil {
			return fmt.Errorf("find compaction boundary: %w", err)
		}
		if keepStartID == 0 {
			return nil
		}
	} else {
		if err := tx.QueryRowContext(ctx, `
			SELECT id
			FROM messages
			WHERE chat_id = ? AND archived = 0 AND role = 'user'
			ORDER BY id DESC
			LIMIT 1 OFFSET ?
		`, chatID, keepUserTurns-1).Scan(&keepStartID); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return nil
			}
			return fmt.Errorf("find compaction boundary: %w", err)
		}
	}

	rows, err := tx.QueryContext(ctx, `
		SELECT m.id
		FROM messages m
		WHERE m.chat_id = ? AND m.archived = 0 AND (
			m.id < ?
			OR m.tool_name = ?
			OR EXISTS (
				SELECT 1 FROM tool_calls tc
				WHERE tc.message_id = m.id AND tc.function_name = ?
			)
		)
		ORDER BY id ASC
	`, chatID, keepStartID, agent.CompactionToolName, agent.CompactionToolName)
	if err != nil {
		return fmt.Errorf("list archived messages: %w", err)
	}
	var archivedIDs []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return fmt.Errorf("scan archived message id: %w", err)
		}
		archivedIDs = append(archivedIDs, id)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return fmt.Errorf("read archived message ids: %w", err)
	}
	rows.Close()
	if len(archivedIDs) == 0 {
		return nil
	}

	idsJSON, err := json.Marshal(archivedIDs)
	if err != nil {
		return fmt.Errorf("marshal archived message ids: %w", err)
	}

	if _, err := tx.ExecContext(ctx, `
		INSERT INTO compactions (chat_id, summary, archived_message_ids, created_at)
		VALUES (?, ?, ?, ?)
	`, chatID, summary, string(idsJSON), time.Now()); err != nil {
		return fmt.Errorf("insert compaction: %w", err)
	}

	if _, err := tx.ExecContext(ctx, `
		UPDATE messages
		SET archived = 1
			WHERE chat_id = ? AND archived = 0 AND (
				id < ?
				OR tool_name = ?
				OR EXISTS (
					SELECT 1 FROM tool_calls
					WHERE tool_calls.message_id = messages.id AND tool_calls.function_name = ?
				)
			)
		`, chatID, keepStartID, agent.CompactionToolName, agent.CompactionToolName); err != nil {
		return fmt.Errorf("archive messages: %w", err)
	}

	for _, msg := range agent.CompactionSummaryMessages(summary, continueTask) {
		messageID, err := insertAgentMessage(ctx, tx, chatID, msg, "")
		if err != nil {
			return err
		}
		for _, toolCall := range msg.ToolCalls {
			if err := insertAgentToolCall(ctx, tx, messageID, toolCall); err != nil {
				return err
			}
		}
	}

	return tx.Commit()
}

func parseAgentSQLiteTime(value string) (time.Time, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return time.Time{}, nil
	}
	for _, layout := range []string{
		time.RFC3339Nano,
		"2006-01-02 15:04:05.999999999-07:00",
		"2006-01-02 15:04:05.999999999Z07:00",
		"2006-01-02 15:04:05.999999999",
		"2006-01-02 15:04:05-07:00",
		"2006-01-02 15:04:05Z07:00",
		"2006-01-02 15:04:05",
	} {
		t, err := time.Parse(layout, value)
		if err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("unsupported time format %q", value)
}

func latestAgentModelForChat(ctx context.Context, db *sql.DB, chatID string) (string, error) {
	var modelName string
	if err := db.QueryRowContext(ctx, `
		SELECT model_name
		FROM messages
		WHERE chat_id = ? AND model_name IS NOT NULL AND model_name != ''
		ORDER BY updated_at DESC, id DESC
		LIMIT 1
	`, chatID).Scan(&modelName); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return "", nil
		}
		return "", err
	}
	return modelName, nil
}

func currentAgentModelSelectExpr(chatAlias string) string {
	return fmt.Sprintf(`COALESCE(
				NULLIF(%[1]s.model_name, ''),
				(
					SELECT lm.model_name
					FROM messages lm
					WHERE lm.chat_id = %[1]s.id AND lm.model_name IS NOT NULL AND lm.model_name != ''
					ORDER BY lm.updated_at DESC, lm.id DESC
					LIMIT 1
				)
			)`, chatAlias)
}

func messagesContainCompactionSummary(messages []api.Message) bool {
	for _, msg := range messages {
		if msg.Role == "tool" && msg.ToolName == agent.CompactionToolName && strings.HasPrefix(msg.Content, agent.CompactionSummaryMessagePrefix) {
			return true
		}
		if (msg.Role == "user" || msg.Role == "system") && strings.HasPrefix(msg.Content, agent.CompactionSummaryMessagePrefix) {
			return true
		}
	}
	return false
}

func moveCompactionSummaryBeforeKeptMessages(messages []api.Message) []api.Message {
	start := -1
	end := -1
	for i, msg := range messages {
		if msg.Role == "assistant" {
			for _, call := range msg.ToolCalls {
				if call.Function.Name == agent.CompactionToolName {
					start = i
					end = i + 1
					if end < len(messages) && messages[end].Role == "tool" && messages[end].ToolName == agent.CompactionToolName {
						end++
					}
					break
				}
			}
		}
		if start >= 0 {
			break
		}
	}
	if start <= 0 || end <= start {
		return messages
	}

	insertAt := leadingSystemMessageCount(messages[:start])
	reordered := make([]api.Message, 0, len(messages))
	reordered = append(reordered, messages[:insertAt]...)
	reordered = append(reordered, messages[start:end]...)
	reordered = append(reordered, messages[insertAt:start]...)
	reordered = append(reordered, messages[end:]...)
	return reordered
}

func insertCompactionSummaryAfterLeadingSystemMessages(messages, summary []api.Message) []api.Message {
	insertAt := leadingSystemMessageCount(messages)
	reordered := make([]api.Message, 0, len(messages)+len(summary))
	reordered = append(reordered, messages[:insertAt]...)
	reordered = append(reordered, summary...)
	reordered = append(reordered, messages[insertAt:]...)
	return reordered
}

func leadingSystemMessageCount(messages []api.Message) int {
	for i, msg := range messages {
		if msg.Role != "system" {
			return i
		}
	}
	return len(messages)
}

func insertAgentMessage(ctx context.Context, tx *sql.Tx, chatID string, msg api.Message, model string) (int64, error) {
	now := time.Now()
	modelName := sql.NullString{}
	if model != "" {
		modelName = sql.NullString{String: model, Valid: true}
	}
	imagesJSON, err := marshalAgentMessageImages(msg.Images)
	if err != nil {
		return 0, err
	}
	result, err := tx.ExecContext(ctx, `
		INSERT INTO messages (chat_id, role, content, thinking, images, tool_name, tool_call_id, model_name, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`, chatID, msg.Role, msg.Content, msg.Thinking, imagesJSON, msg.ToolName, msg.ToolCallID, modelName, now, now)
	if err != nil {
		return 0, fmt.Errorf("insert message: %w", err)
	}
	id, err := result.LastInsertId()
	if err != nil {
		return 0, fmt.Errorf("get message id: %w", err)
	}
	return id, nil
}

func marshalAgentMessageImages(images []api.ImageData) (string, error) {
	if len(images) == 0 {
		return "[]", nil
	}
	data, err := json.Marshal(images)
	if err != nil {
		return "", fmt.Errorf("marshal message images: %w", err)
	}
	return string(data), nil
}

func unmarshalAgentMessageImages(value string) ([]api.ImageData, error) {
	value = strings.TrimSpace(value)
	if value == "" || value == "null" {
		return nil, nil
	}
	var images []api.ImageData
	if err := json.Unmarshal([]byte(value), &images); err != nil {
		return nil, fmt.Errorf("unmarshal message images: %w", err)
	}
	return images, nil
}

func insertAgentToolCall(ctx context.Context, tx *sql.Tx, messageID int64, call api.ToolCall) error {
	args, err := json.Marshal(call.Function.Arguments)
	if err != nil {
		return fmt.Errorf("marshal tool arguments: %w", err)
	}
	_, err = tx.ExecContext(ctx, `
		INSERT INTO tool_calls (message_id, type, tool_call_id, function_name, function_arguments)
		VALUES (?, ?, ?, ?, ?)
	`, messageID, "function", call.ID, call.Function.Name, string(args))
	if err != nil {
		return fmt.Errorf("insert tool call: %w", err)
	}
	return nil
}

func getAgentToolCalls(ctx context.Context, db *sql.DB, messageID int64) ([]api.ToolCall, error) {
	rows, err := db.QueryContext(ctx, `
		SELECT tool_call_id, function_name, function_arguments FROM tool_calls WHERE message_id = ? ORDER BY id ASC
	`, messageID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var calls []api.ToolCall
	for rows.Next() {
		var id, name, argsJSON string
		if err := rows.Scan(&id, &name, &argsJSON); err != nil {
			return nil, err
		}
		var args api.ToolCallFunctionArguments
		if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
			return nil, err
		}
		calls = append(calls, api.ToolCall{
			ID: id,
			Function: api.ToolCallFunction{
				Name:      name,
				Arguments: args,
			},
		})
	}
	return calls, rows.Err()
}

func latestCompactionSummary(ctx context.Context, db *sql.DB, chatID string) (string, error) {
	var summary string
	if err := db.QueryRowContext(ctx, `
		SELECT summary
		FROM compactions
		WHERE chat_id = ?
		ORDER BY id DESC
		LIMIT 1
	`, chatID).Scan(&summary); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return "", nil
		}
		return "", fmt.Errorf("get latest compaction summary: %w", err)
	}
	return summary, nil
}

func maybeSetAgentTitle(ctx context.Context, tx *sql.Tx, chatID string, content string) error {
	title := strings.TrimSpace(content)
	if len([]rune(title)) > 64 {
		title = string([]rune(title)[:64])
	}
	_, err := tx.ExecContext(ctx, `
		UPDATE chats
		SET title = CASE WHEN title = '' THEN ? ELSE title END
		WHERE id = ?
	`, title, chatID)
	return err
}
