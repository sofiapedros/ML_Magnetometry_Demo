
"""Entry point for the public Streamlit demo with private Hugging Face assets."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys
import time

import streamlit as st
from huggingface_hub import snapshot_download

from asset_paths import repair_manifest


ROOT = Path(__file__).resolve().parent


def log(message: str) -> None:
    """Write timestamped diagnostic messages to Streamlit Cloud logs."""
    print(
        f"[ML-MAGNETOMETRY] {message}",
        flush=True,
    )


def setting(name: str, default: str = "") -> str:
    """Read from Streamlit secrets or environment variables."""
    value = os.environ.get(name)

    if value is not None:
        return value.strip()

    try:
        return str(st.secrets.get(name, default)).strip()
    except (FileNotFoundError, KeyError, AttributeError):
        return default


@st.cache_resource(
    show_spinner="Downloading private inference assets from Hugging Face…"
)
def download_private_assets(
    repo_id: str,
    token: str,
    revision: str,
) -> dict:

    start = time.perf_counter()

    log("[HF] Starting private assets download...")
    log(f"[HF] Repository: {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        revision=revision or None,
        local_dir=ROOT,
        allow_patterns=[
            "checkpoints/**",
            "dataset/**",
        ],
    )

    elapsed = time.perf_counter() - start
    log(f"[HF] Download completed in {elapsed:.1f} seconds")

    log("[CHECK] Checking checkpoints directory...")

    checkpoints_dir = ROOT / "checkpoints"

    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(
            "No checkpoints/ directory was downloaded."
        )

    checkpoint_files = list(
        checkpoints_dir.rglob("seed_*.pt")
    )

    log(f"[CHECK] Found {len(checkpoint_files)} checkpoints")

    if not checkpoint_files:
        raise FileNotFoundError(
            "No checkpoints/<model>/seed_*.pt files were downloaded."
        )

    log("[DATASET] Repairing Windows paths in ml_manifest.csv...")

    repair_start = time.perf_counter()

    report = repair_manifest(ROOT)

    repair_elapsed = time.perf_counter() - repair_start

    log(
        f"[DATASET] Manifest processed in "
        f"{repair_elapsed:.1f} seconds"
    )

    log(
        f"[DATASET] Unresolved paths: "
        f"{report['unresolved']}"
    )

    log("[ASSETS] Private assets ready")

    return report


# -------------------------------------------------
# APPLICATION STARTUP
# -------------------------------------------------

log("[STARTUP] Starting ML Magnetometry application")

repo_id = setting("HF_REPO_ID")
token = setting("HF_TOKEN")
revision = setting("HF_REVISION")

log(f"[CONFIG] HF_REPO_ID configured: {bool(repo_id)}")
log(f"[CONFIG] HF_TOKEN configured: {bool(token)}")

if not repo_id or not token:
    log("[ERROR] Missing Hugging Face configuration")

    st.error(
        "Missing HF_REPO_ID or HF_TOKEN. "
        "Configure them under Streamlit app settings → Secrets."
    )

    st.stop()


# -------------------------------------------------
# DOWNLOAD PRIVATE MODELS AND DATA
# -------------------------------------------------

try:
    log("[ASSETS] Preparing private assets...")

    asset_report = download_private_assets(
        repo_id,
        token,
        revision,
    )

    log("[ASSETS] Preparation completed successfully")

except Exception as exc:

    log(
        f"[ERROR] Asset preparation failed: "
        f"{type(exc).__name__}"
    )

    st.error(
        "Unable to prepare the private models "
        "and dataset from Hugging Face."
    )

    st.caption(
        f"Problem type: {type(exc).__name__}. "
        "Verify your private repository, token permissions, "
        "manifest and uploaded files."
    )

    st.stop()


# -------------------------------------------------
# CONFIGURE PROJECT ROOT
# -------------------------------------------------

log("[CONFIG] Configuring project paths...")

os.environ["MAGNETIC_PROJECT_ROOT"] = str(ROOT)
os.environ["MAGNETIC_DEPLOYED_DEMO"] = "1"

st.session_state["project_root_input"] = str(ROOT)

os.environ["MAGNETIC_SHOW_TEST_DATASET"] = setting(
    "SHOW_TEST_DATASET",
    "false",
)

for path in (
    ROOT,
    ROOT / "src" / "models",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

log("[CONFIG] Project paths configured")


# -------------------------------------------------
# DATASET WARNINGS
# -------------------------------------------------

if asset_report["unresolved"]:

    log("[WARNING] Some dataset paths are unresolved")

    with st.expander(
        "Dataset path warnings",
        expanded=False,
    ):
        st.warning(
            "Some paths in dataset/ml_manifest.csv "
            "could not be mapped to downloaded files: "
            + ", ".join(
                f"{k}: {v}"
                for k, v in asset_report["unresolved"].items()
            )
            + ". Upload those files to Hugging Face "
            "with their dataset/ subpaths."
        )


# -------------------------------------------------
# LAUNCH ORIGINAL DEMO
# -------------------------------------------------

log("[DEMO] Launching magnetic loop explorer...")

demo_start = time.perf_counter()

try:
    runpy.run_path(
        str(ROOT / "src" / "models" / "demo.py"),
        run_name="__main__",
    )

    elapsed = time.perf_counter() - demo_start

    log(
        f"[DEMO] Script execution completed "
        f"in {elapsed:.1f} seconds"
    )

except Exception as exc:
    log(
        f"[ERROR] Demo execution failed: "
        f"{type(exc).__name__}"
    )
    raise
