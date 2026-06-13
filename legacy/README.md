# Legacy — Streamlit dashboard (archived)

This folder holds the original **Streamlit** dashboard that the project shipped
before the React/web migration. It is kept for reference only; the live UI is
now [`../frontend/`](../frontend/) (React + Vite + Plotly) talking to the
FastAPI backend.

- `dashboard/app.py` — the 19-view Streamlit app
- `streamlit-config/` — the old `.streamlit/` theme config

A fully runnable snapshot (with the matching `requirements.txt` that still
pins `streamlit` + `plotly`) is tagged in git:

```bash
git checkout streamlit-v1     # working Streamlit app + deps
```

The current `requirements.txt` no longer installs Streamlit/Plotly, so this
archived copy will not run as-is against the trimmed dependency set.
