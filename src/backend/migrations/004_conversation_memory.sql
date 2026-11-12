-- Conversation memory for context engineering (app/harness/memory.py).
-- The API also adds these columns at startup (init_db); this file is for manual runs.
ALTER TABLE agent_conversations
  ADD COLUMN IF NOT EXISTS kind VARCHAR(32) NOT NULL DEFAULT 'agent';
ALTER TABLE agent_conversations
  ADD COLUMN IF NOT EXISTS context_summary TEXT;
ALTER TABLE agent_conversations
  ADD COLUMN IF NOT EXISTS summarized_turns INTEGER NOT NULL DEFAULT 0;
ALTER TABLE agent_conversations
  ADD COLUMN IF NOT EXISTS context_state JSON;
