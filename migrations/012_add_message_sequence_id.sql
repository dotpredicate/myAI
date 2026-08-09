ALTER TABLE messages ADD COLUMN sequence_id INTEGER;

WITH numbered AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY conversation_id
            ORDER BY created_at ASC, id ASC
        ) AS seq
    FROM messages
)
UPDATE messages
SET sequence_id = numbered.seq
FROM numbered
WHERE messages.id = numbered.id;

ALTER TABLE messages ALTER COLUMN sequence_id SET NOT NULL;

CREATE UNIQUE INDEX messages_conversation_sequence_uidx
ON messages(conversation_id, sequence_id);