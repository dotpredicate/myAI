ALTER TABLE agents ADD COLUMN spawnable_agents JSONB NOT NULL DEFAULT '[]'::jsonb;
