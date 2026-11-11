"""Bronze layer: land the raw source exactly as received, plus lineage metadata.

No type casting or cleaning happens here. Every value is kept as a string so the
bronze table is a faithful, replayable copy of the source file.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)

BRONZE_TABLE = "load_profiles_raw.parquet"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest(source: Path = config.RAW_FILE, out_dir: Path = config.BRONZE_DIR) -> pd.DataFrame:
    if not source.exists():
        raise FileNotFoundError(f"Raw file not found: {source}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    digest = _sha256(source)

    df["_source_file"] = source.name
    df["_source_sha256"] = digest
    df["_source_row"] = range(2, len(df) + 2)  # 1-based, header is row 1
    df["_ingested_at"] = ingested_at

    df.to_parquet(out_dir / BRONZE_TABLE, index=False)

    manifest = {
        "table": BRONZE_TABLE,
        "source_file": source.name,
        "source_sha256": digest,
        "rows": int(len(df)),
        "source_columns": [c for c in df.columns if not c.startswith("_")],
        "ingested_at": ingested_at,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Bronze: %s rows landed from %s", len(df), source.name)
    return df


def read() -> pd.DataFrame:
    return pd.read_parquet(config.BRONZE_DIR / BRONZE_TABLE)
