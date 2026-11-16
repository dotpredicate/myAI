CREATE TABLE IF NOT EXISTS benchmark_results (
    model_id TEXT NOT NULL,
    quant TEXT NOT NULL,
    depth INTEGER NOT NULL,
    tg_tps REAL,
    pp_tps REAL,
    vram_peak_bytes INTEGER,
    duration_ms INTEGER,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_id, quant, depth)
);
