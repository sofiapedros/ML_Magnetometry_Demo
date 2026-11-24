# ML Magnetometry — Interactive Demo

An interactive **Streamlit demonstration for machine-learning classification of magnetic states from magnetic hysteresis loops**. The interface loads real trained model checkpoints, visualizes predictions, and supports uploading magnetic measurements. It is the deployment-only companion to the ML Magnetometry research project, not the training or simulation pipeline.

## What the app does

- Classifies magnetic states as **SPM**, **SD**, or **MD** using saved machine-learning checkpoints.
- Supports the model families registered in the project, including MLP, RNN and physics-/Preisach-enhanced variants, when their checkpoints are provided.
- Offers ensemble predictions across available training seeds, plots and model interpretation tools.
- Accepts uploaded MPMS3 `.dat` files or a compatible measurement `.csv` for exploratory inference. Experimental measurements may differ from the simulation training distribution; predictions should be interpreted cautiously.
- Can optionally display test-dataset examples. **This is disabled by default** to avoid exposing private research samples through the public demo.

The application never generates synthetic predictions when trained models are missing.

## Architecture and privacy

```text
Public GitHub repo (this repository)
  app.py                  -> downloads private assets, prepares paths, launches Streamlit
  asset_paths.py          -> repairs paths in the downloaded manifest
  src/                    -> inference and visualization source code
  requirements.txt
  .python-version         -> Python 3.11 locally

Private Hugging Face *dataset* repo (NOT this GitHub repo)
  checkpoints/<model_name>/seed_0.pt
  checkpoints/<model_name>/seed_1.pt       (optional)
  dataset/ml_manifest.csv
  dataset/sample_..._hysteresis.csv       (for relevant samples)
  dataset/images/*.npy                    (if image-based models need them)
  dataset/other auxiliary files          (if required by your workflow)
```

Only the code goes to **public GitHub**. Checkpoints and the manifest/files go to a **private Hugging Face Dataset repository**, accessible through a restricted token stored in **Streamlit Secrets**. The public app can still reveal *predictions, visualizations, or data* it chooses to display; a private backend does not make the public UI confidential.

### Important: manifests created on Windows

The original `dataset/ml_manifest.csv` may contain paths such as:

```text
C:\Users\User\Documents\...\dataset\sample_d_100nm_..._hysteresis.csv
```

Those paths cannot work on a Linux server. At startup, `asset_paths.py` **repairs the downloaded copy** of the manifest, resolving each path to an actual file under the downloaded `dataset/` directory. It handles Windows absolute paths, repository-relative paths and unique filenames. For example, the value above maps to a file such as:

```text
/mount/src/ml-magnetometry-demo/dataset/sample_d_100nm_..._hysteresis.csv
```

This adjustment does **not** modify the manifest stored on Hugging Face. Upload the actual files corresponding to rows that the demo needs. Any missing or ambiguous filenames will be reported in the app as dataset path warnings; the application cannot reconstruct a file that was not uploaded.

**Keep filenames unique** in the downloaded dataset when possible. If two files share the same filename, preserve their `dataset/` subdirectories in the manifest for unambiguous mapping.

Some features (including parts of the explanation and reference-based inference) read `train` samples, even when visitors upload their own measurements. You may therefore need a small representative *training* subset with valid manifest entries, not just checkpoints. Features requiring files that were not uploaded will not run.

## 1. Upload private assets to Hugging Face

1. Create a **private Dataset repository**: https://huggingface.co/new-dataset
2. Prepare an `hf_assets/` directory with the desired `checkpoints/` and `dataset/` subdirectories as shown above. Do not upload your full 20 GB research archive unless needed by the demo.
3. Upload the folder, preserving its structure. With the Hugging Face CLI installed and authenticated:

   ```bash
   uv tool install huggingface_hub
   hf auth login
   hf upload USERNAME/ML-Magnetometry-Private ./hf_assets . --repo-type dataset
   ```

4. Generate a **fine-grained read-only** token with permission to read that private repository: https://huggingface.co/settings/tokens

## 2. Deploy on Streamlit Community Cloud

1. Push **only the contents of this ZIP** to your public GitHub demo repository. Do not commit `dataset/`, `checkpoints/`, `.env`, tokens or `.streamlit/secrets.toml`.
2. Create an app at https://share.streamlit.io with main file **`app.py`**.
3. In **Advanced settings**, select **Python 3.11**. This is important: the pinned `torch==2.5.1+cpu` does not support Python 3.14. If the existing app runs Python 3.14, recreate the Streamlit deployment using Python 3.11; `.python-version` is a local hint and does not reliably select the cloud runtime.
4. Configure the app's **Secrets** (TOML):

   ```toml
   HF_REPO_ID = "USERNAME/ML-Magnetometry-Private"
   HF_TOKEN = "hf_READ_ONLY_TOKEN"
   # Optional: lock assets to a given Hugging Face commit
   # HF_REVISION = "commit_hash"
   # Optional: only enable if you have permission to display test samples
   # SHOW_TEST_DATASET = "true"
   ```

5. Deploy or reboot. The app downloads private assets using `huggingface_hub.snapshot_download`, repairs the local manifest, then starts the actual inference interface.

The private token is never required in GitHub. Do not paste real tokens into issues, screenshots, or logs.

## 3. Run locally with uv

Use Python 3.11. For a project-isolated environment:

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

Set `HF_REPO_ID` and `HF_TOKEN` as environment variables, then run:

```bash
uv run --no-project streamlit run app.py
```

On PowerShell, assign environment variables using `$env:HF_REPO_ID="..."` and `$env:HF_TOKEN="..."`.

## Troubleshooting

- **Dependency resolver reports `cp314` and `torch==2.5.1+cpu`:** deploy with **Python 3.11**.
- **Missing secrets or download failure:** check `HF_REPO_ID`, fine-grained token permissions, repository visibility and Hugging Face filenames.
- **No checkpoints:** at least one file like `checkpoints/mlp/seed_0.pt` must exist in the private Dataset repository.
- **Dataset path warnings / file-not-found:** upload the missing curve/image files with matching relative paths; the Windows path in the CSV is not a real server file.
- **Model errors when uploading measurements:** certain inference and explanation methods need training reference samples; include the relevant train rows and files from the dataset.
- **Test dataset not visible:** intentionally disabled by default for privacy. Set `SHOW_TEST_DATASET = "true"` only when appropriate and include test files.

## Research disclaimer

This is a research demonstrator, not a validated diagnostic tool. Inference on uploaded measurements is exploratory, particularly when measurement temperatures, units or protocols differ from training data. Model confidence is not a substitute for experimental validation.
