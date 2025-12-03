-- This is the version 2 schema for the app database, the first released schema to users.
-- Do not modify this file. It is used to test that the database schema stays in a consistent state between schema migrations.

CREATE TABLE IF NOT EXISTS settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    device_id TEXT NOT NULL DEFAULT '',
    has_completed_first_run BOOLEAN NOT NULL DEFAULT 0,
    expose BOOLEAN NOT NULL DEFAULT 0,
    survey BOOLEAN NOT NULL DEFAULT TRUE,
    browser BOOLEAN NOT NULL DEFAULT 0,
    models TEXT NOT NULL DEFAULT '',
    remote TEXT NOT NULL DEFAULT '',
    agent BOOLEAN NOT NULL DEFAULT 0,
    tools BOOLEAN NOT NULL DEFAULT 0,
    working_dir TEXT NOT NULL DEFAULT '',
    context_length INTEGER NOT NULL DEFAULT 4096,
    window_width INTEGER NOT NULL DEFAULT 0,
    window_height INTEGER NOT NULL DEFAULT 0,
    config_migrated BOOLEAN NOT NULL DEFAULT 0,
    schema_version INTEGER NOT NULL DEFAULT 2
);

-- Insert default settings row if it doesn't exist
INSERT OR IGNORE INTO settings (id) VALUES (1);

CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    thinking TEXT NOT NULL DEFAULT '',
    stream BOOLEAN NOT NULL DEFAULT 0,
    model_name TEXT,
    model_cloud BOOLEAN,
    model_ollama_host BOOLEAN,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    thinking_time_start TIMESTAMP,
    thinking_time_end TIMESTAMP,
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    type TEXT NOT NULL,
    function_name TEXT NOT NULL,
    function_arguments TEXT NOT NULL,
    function_result TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_message_id ON tool_calls(message_id);
