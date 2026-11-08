-- The kind column originally allowed only external providers. The built-in
-- llama.cpp provider now lives in the same table too, so widen the check and
-- seed its configuration row. Its instance is still created in code because it
-- owns the model server lifecycle.

ALTER TABLE inference_providers DROP CONSTRAINT inference_providers_kind_check;

ALTER TABLE inference_providers
    ADD CONSTRAINT inference_providers_kind_check
    CHECK (kind IN ('llama_cpp', 'openai_compatible'));

INSERT INTO inference_providers
    (key, display_name, description, kind, base_url, api_key, models,
     timeout_seconds, headers, extra_body, include_reasoning_content)
VALUES
    ('llama.cpp', 'Llama.cpp Server', 'Embedded local llama.cpp server',
     'llama_cpp', '', NULL, '[]', 120, '{}', '{}', FALSE);
