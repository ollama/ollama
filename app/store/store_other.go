//go:build !windows && !darwin

package store

import (
	"context"
	"database/sql"
	"errors"
	"time"

	"github.com/ollama/ollama/api"
)

var errStoreUnavailable = errors.New("app store is only available on Windows and macOS")

type Store struct{}

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

func New(string) (*Store, error) {
	return nil, errStoreUnavailable
}

func (s *Store) Close() error {
	return nil
}

func (s *Store) EnsureChat(context.Context, string, string) error {
	return errStoreUnavailable
}

func (s *Store) SetChatModel(context.Context, string, string) error {
	return errStoreUnavailable
}

func (s *Store) AppendAgentMessage(context.Context, string, api.Message, string) error {
	return errStoreUnavailable
}

func (s *Store) UpdateLastAgentMessage(context.Context, string, api.Message, string) error {
	return errStoreUnavailable
}

func (s *Store) AgentChat(context.Context, string) (*AgentChat, error) {
	return nil, sql.ErrNoRows
}

func (s *Store) LatestChat(context.Context) (*AgentChat, error) {
	return nil, sql.ErrNoRows
}

func (s *Store) LatestChatForModel(context.Context, string) (*AgentChat, error) {
	return nil, sql.ErrNoRows
}

func (s *Store) ListChats(context.Context, int) ([]ChatSummary, error) {
	return nil, errStoreUnavailable
}

func (s *Store) ListUserMessages(context.Context, int) ([]string, error) {
	return nil, errStoreUnavailable
}

func (s *Store) ArchiveForCompaction(context.Context, string, int, string, bool) error {
	return errStoreUnavailable
}
