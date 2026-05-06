"""
adhoc_data_processing.py — local stub for the adhoc_data_processing library.

In environments where the real library is installed this file is never loaded.
Locally it provides adp.run() and adp.set_pipeline_state() so the script runs
without modification.

Manifests configure where data lives and which MODEL_MODE to use:
  input_manifest[0]['model_mode']  "GENERATE" | "LOAD"
"""

import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# Local manifest defaults
# ─────────────────────────────────────────────────────────────────────────────

INPUT_MANIFEST: List[Dict[str, Any]] = [
    {
        "prefix":      "",               # no subfolder prefix locally
        "bucket_name": "vehiclebucket",
        "local_path":  ".",
        "model_mode":  "GENERATE",       # "GENERATE" | "LOAD"
    }
]

OUTPUT_MANIFEST: List[Dict[str, Any]] = [
    {
        "local_path":  ".",
        "bucket_path": "",
        "bucket_name": "mlbucket",
    }
]

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline state — persisted to a local file between runs
# ─────────────────────────────────────────────────────────────────────────────

_STATE_FILE = Path(".adp_pipeline_state")


def set_pipeline_state(state: str) -> None:
    _STATE_FILE.write_text(state)
    print(f"[adp] pipeline_state → {state}")


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

def run(main_fn: Callable, globals_dict: Dict[str, Any], flag: bool) -> Any:
    pipeline_state = _STATE_FILE.read_text().strip() if _STATE_FILE.exists() else ""
    job_id = str(uuid.uuid4())[:8]
    print(f"[adp] run  job_id={job_id}  pipeline_state={pipeline_state!r}  model_mode={INPUT_MANIFEST[0]['model_mode']!r}")
    return main_fn(INPUT_MANIFEST, OUTPUT_MANIFEST, pipeline_state, job_id)
