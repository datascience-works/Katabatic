# Katabatic — Katabatic-Manya-Dashboard

## Run all four pages on one port

From the repository root:

```bash
# If npm is not found in your terminal:
source ~/.nvm/nvm.sh

npm --prefix frontend/results-dashboard run dev
```

Open **http://localhost:5173**. Only one server and one terminal are required.
`npm run dev` from any of the four frontend directories starts this same workspace.
The earlier `npm --prefix frontend/Katabatic-Manya-Dashboard run dev:workspace`
command also starts the single server.

| Page | Route |
| --- | --- |
| Manya dashboard | `/` |
| Upload dataset | `/datasets` |
| Model configuration | `/models` |
| Results dashboard | `/results` |

Sidebar links and workflow actions stay on port 5173. Direct links, refresh,
and browser Back/Forward work on each route. Stop the server with Ctrl+C.
If port 5173 is already occupied, stop the previous workspace server first.

## Setup and production build

Dependencies are already installed in this workspace. On a fresh checkout,
run `npm install` inside each of the four frontend directories first.
To build and preview the complete frontend:

```bash
npm --prefix frontend/results-dashboard run build
npm --prefix frontend/results-dashboard run preview
```

The combined output is `frontend/results-dashboard/dist`. A static deployment
must serve `index.html` for `/datasets`, `/models`, and `/results` as well as `/`.

## Implementation

`results-dashboard/src/WorkspaceApp.jsx` selects the page by URL and imports
its existing implementation from the original directory. Page CSS is applied
only for the current route so styles do not leak between designs. Vite uses a
single React instance across the four directories. No extra directory or
proxy server is needed.

The overview panels and centered upload form retain their structure.
Configuration keeps its workflow, form and summary; Results keeps its evaluation,
comparison and download sections. Navigation is defined in each app's local
`navigation` module using paths on the same origin.

This is a frontend preview: selected files are not uploaded, training is not run,
and Results shows sample data. The previous Next.js API handlers are not served
by this Vite preview; the dashboard continues to use its existing sample data.
