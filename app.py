"""Entry point for the public Streamlit demo with private Hugging Face assets."""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

import streamlit as st
from huggingface_hub import snapshot_download

from asset_paths import repair_manifest

ROOT = Path(__file__).resolve().parent


def setting(name: str, default: str = "") -> str:
    """Read from Streamlit secrets or environment variables (for local runs)."""
    value = os.environ.get(name)
    if value is not None:
        return value.strip()
    try:
        return str(st.secrets.get(name, default)).strip()
    except (FileNotFoundError, KeyError, AttributeError):
        return default


@st.cache_resource(show_spinner="Downloading private inference assets from Hugging Face…")
def download_private_assets(repo_id: str, token: str, revision: str) -> dict:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        revision=revision or None,
        local_dir=ROOT,
        allow_patterns=["checkpoints/**", "dataset/**"],
    )
    if not (ROOT / "checkpoints").is_dir():
        raise FileNotFoundError("No checkpoints/ directory was downloaded.")
    if not any((ROOT / "checkpoints").rglob("seed_*.pt")):
        raise FileNotFoundError("No checkpoints/<model>/seed_*.pt files were downloaded.")
    return repair_manifest(ROOT)


repo_id = setting("HF_REPO_ID")
token = setting("HF_TOKEN")
revision = setting("HF_REVISION")
if not repo_id or not token:
    st.error("Missing HF_REPO_ID or HF_TOKEN. Configure them under Streamlit app settings → Secrets.")
    st.stop()

try:
    asset_report = download_private_assets(repo_id, token, revision)
except Exception as exc:
    st.error("Unable to prepare the private models and dataset from Hugging Face.")
    st.caption(f"Problem type: {type(exc).__name__}. Verify your private repository, token permissions, manifest and uploaded files.")
    # Avoid echoing API responses that might contain sensitive file paths or credentials.
    st.stop()

os.environ["MAGNETIC_PROJECT_ROOT"] = str(ROOT)
os.environ["MAGNETIC_DEPLOYED_DEMO"] = "1"
# A hosted Streamlit app should never request a local project path from visitors.
st.session_state["project_root_input"] = str(ROOT)
os.environ["MAGNETIC_SHOW_TEST_DATASET"] = setting("SHOW_TEST_DATASET", "false")
for path in (ROOT, ROOT / "src" / "models"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if asset_report["unresolved"]:
    with st.expander("Dataset path warnings", expanded=False):
        st.warning(
            "Some paths in dataset/ml_manifest.csv could not be mapped to downloaded files: "
            + ", ".join(f"{k}: {v}" for k, v in asset_report["unresolved"].items())
            + ". Upload those files to Hugging Face with their dataset/ subpaths."
        )

runpy.run_path(str(ROOT / "src" / "models" / "demo.py"), run_name="__main__")
