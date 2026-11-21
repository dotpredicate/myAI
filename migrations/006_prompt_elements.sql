-- The prompt is the user's element: `Message` is only what the assistant said,
-- so the row no longer needs a role to say who spoke.
UPDATE messages
SET elements = json_set(elements, '$.type', 'prompt')
WHERE role = 'user'
  AND json_extract(elements, '$.type') = 'message';

ALTER TABLE messages DROP COLUMN role;
