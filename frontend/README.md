# Frontend

The complete frontend runs as one Vite application on port `5173`, starting at
the main dashboard.

## Run the frontend

From the repository root, install the workspace dependencies once:

```bash
npm --prefix frontend/main-dashboard install
```

Start the workspace through the main dashboard:

```bash
npm --prefix frontend/main-dashboard run dev
```

When your terminal is already in the `frontend` directory, use:

```bash
npm --prefix main-dashboard run dev
```

Open http://localhost:5173. It starts at the main dashboard (`/`).

| Page | Route |
| --- | --- |
| Main dashboard | `/` |
| Upload dataset | `/datasets` |
| Model configuration | `/models` |
| Results dashboard | `/results` |

Stop the server with `Ctrl+C`. Port `5173` must be available before starting.
