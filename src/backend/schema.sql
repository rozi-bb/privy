-- App tables. Applied on every backend start, so every statement must be
-- safe to run again. The GDPR chunks (pgvector) and the conversation
-- checkpoints live in their own tables, created by langchain-postgres and
-- the LangGraph checkpointer.

CREATE TABLE IF NOT EXISTS users (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    email text NOT NULL UNIQUE,
    password_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- A thread's id is also the checkpointer's thread_id, so owning the row is
-- what grants access to the conversation stored under it.
CREATE TABLE IF NOT EXISTS threads (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    title text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS threads_user_id_idx ON threads (user_id, created_at);

-- One row per answered question. Feeds usage limits and cost per user later.
CREATE TABLE IF NOT EXISTS usage (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    thread_id uuid NOT NULL REFERENCES threads (id) ON DELETE CASCADE,
    model text NOT NULL,
    input_tokens integer NOT NULL,
    output_tokens integer NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS usage_user_id_idx ON usage (user_id, created_at);
