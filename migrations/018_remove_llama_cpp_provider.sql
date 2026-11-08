-- The built-in llama.cpp provider is registered in code and no longer stored
-- in the table. External providers are the only rows, so the kind column is
-- redundant.

DELETE FROM inference_providers WHERE key = 'llama.cpp';

ALTER TABLE inference_providers DROP COLUMN kind;
