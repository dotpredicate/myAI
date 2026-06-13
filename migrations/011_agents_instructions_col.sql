ALTER TABLE agents ADD COLUMN instructions TEXT;
UPDATE agents SET instructions='';
ALTER TABLE agents ALTER COLUMN instructions SET NOT NULL;