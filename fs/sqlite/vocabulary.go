// vocabulary.go provides vocabulary loading from SQLite for model initialization.
package sqlite

import (
	"encoding/json"
	"fmt"

	"github.com/ollama/ollama/model"
)

// LoadVocabulary loads a complete Vocabulary struct from SQLite database.
// This is designed to be a drop-in replacement for loading from GGUF metadata.
func (m *Model) LoadVocabulary() (*model.Vocabulary, error) {
	vocab := m.db.GetVocabulary()

	// Get all token strings
	tokens, err := vocab.GetAllTokens()
	if err != nil {
		return nil, fmt.Errorf("failed to load tokens: %w", err)
	}

	if len(tokens) == 0 {
		return nil, fmt.Errorf("vocabulary is empty")
	}

	// Load scores from metadata (JSON array)
	scores := m.KV().Floats("tokenizer.ggml.scores")

	// Load token types from metadata (JSON array)
	types := m.KV().Ints("tokenizer.ggml.token_type")

	// Load BPE merges from metadata (JSON array)
	merges := m.KV().Strings("tokenizer.ggml.merges")

	// Load special token settings
	addBOS := m.KV().Bool("tokenizer.ggml.add_bos_token", true)
	addEOS := m.KV().Bool("tokenizer.ggml.add_eos_token", false)

	bosID := int32(m.KV().Uint("tokenizer.ggml.bos_token_id"))
	eosID := int32(m.KV().Uint("tokenizer.ggml.eos_token_id"))

	// Load additional EOS tokens if present
	eosIDs := []int32{eosID}
	if additionalEOS := m.KV().Ints("tokenizer.ggml.eos_token_ids"); len(additionalEOS) > 0 {
		eosIDs = append(eosIDs, additionalEOS...)
	}

	return &model.Vocabulary{
		Values: tokens,
		Scores: scores,
		Types:  types,
		Merges: merges,
		AddBOS: addBOS,
		AddEOS: addEOS,
		BOS:    []int32{bosID},
		EOS:    eosIDs,
	}, nil
}

// LoadVocabularyMinimal loads just the token strings for simple lookups.
func (m *Model) LoadVocabularyMinimal() (*model.Vocabulary, error) {
	vocab := m.db.GetVocabulary()

	tokens, err := vocab.GetAllTokens()
	if err != nil {
		return nil, fmt.Errorf("failed to load tokens: %w", err)
	}

	return &model.Vocabulary{
		Values: tokens,
	}, nil
}

// SaveVocabulary saves a vocabulary to SQLite.
// Used for model conversion and training updates.
func (m *Model) SaveVocabulary(vocab *model.Vocabulary) error {
	tx, err := m.db.BeginTransaction()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	// Clear existing vocab
	if _, err := tx.Exec("DELETE FROM vocab"); err != nil {
		return err
	}

	// Insert all tokens
	stmt, err := tx.Prepare("INSERT INTO vocab (token_id, token_string) VALUES (?, ?)")
	if err != nil {
		return err
	}
	defer stmt.Close()

	for i, token := range vocab.Values {
		if _, err := stmt.Exec(i, token); err != nil {
			return err
		}
	}

	// Save scores to metadata
	if len(vocab.Scores) > 0 {
		scoresJSON, _ := json.Marshal(vocab.Scores)
		if _, err := tx.Exec(
			"INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
			"tokenizer.ggml.scores", string(scoresJSON),
		); err != nil {
			return err
		}
	}

	// Save types to metadata
	if len(vocab.Types) > 0 {
		typesJSON, _ := json.Marshal(vocab.Types)
		if _, err := tx.Exec(
			"INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
			"tokenizer.ggml.token_type", string(typesJSON),
		); err != nil {
			return err
		}
	}

	// Save merges to metadata
	if len(vocab.Merges) > 0 {
		mergesJSON, _ := json.Marshal(vocab.Merges)
		if _, err := tx.Exec(
			"INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
			"tokenizer.ggml.merges", string(mergesJSON),
		); err != nil {
			return err
		}
	}

	return tx.Commit()
}

// TokenEncode returns the token ID for a string directly from SQLite.
// This bypasses the in-memory vocabulary for memory-efficient lookups.
func (m *Model) TokenEncode(s string) (int32, error) {
	vocab := m.db.GetVocabulary()
	id, err := vocab.GetTokenID(s)
	if err != nil {
		return -1, err
	}
	return int32(id), nil
}

// TokenDecode returns the string for a token ID directly from SQLite.
func (m *Model) TokenDecode(id int32) (string, error) {
	vocab := m.db.GetVocabulary()
	return vocab.GetTokenString(int(id))
}
