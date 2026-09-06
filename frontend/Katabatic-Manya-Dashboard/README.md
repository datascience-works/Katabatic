# Manya's Katabatic dashboard

This folder contains an independent Next.js, React and TypeScript implementation of the Katabatic overview dashboard.

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## API endpoints

- `GET /api/datasets`
- `GET /api/experiments` (optional `?status=completed` filter)
- `GET /api/models`
- `GET /api/activity`

The current endpoints use typed representative data so the UI can be integrated immediately. Replace the arrays in `lib/dashboard-data.ts` with calls to the existing Katabatic backend when its final service contract is confirmed.
