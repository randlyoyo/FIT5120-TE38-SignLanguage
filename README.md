# FIT5120-TE38-SignLanguage

FIT5120 TE38 team's formal capstone project — an Inclusive Auslan Learning Assistant
(see `Inclusive_Auslan_Learning_Assistant_Proposal.pptx`).

## `website/` — HandMirror (main app)

**Live:** https://handmirror.vercel.app (client, on Vercel) — talks to
https://handmirror-server-production.up.railway.app (API, on Railway), backed
by a shared MySQL database also hosted on Railway.

A full-stack Auslan sign-language reference, catalogued like a field
specimen collection: search or browse signs by keyword, synonym, or
category; open a sign to see its demonstration and usage notes; mark signs
as learned (tracked client-side in `localStorage`, no account required).

Covers Epic 1 (US1.1–1.4: search, tag browsing, sign detail, pagination) and
Epic 2 (US2.1–2.4: demonstration, playback controls, text description, mark
as learned) of the project's user stories.

**Stack:** React + TypeScript + Vite (`website/client`), Express + MySQL
(`website/server`, via `mysql2`).

**Structure:**
- `website/client` — the frontend (pages, components, API client, hooks)
- `website/server` — the Express API and database access layer
- `website/docs` — internal reading guides for the search/browse and
  detail/demonstration code paths (onboarding reference for the team)

**Run locally:**
```
cd website
npm run install:all
npm run seed         # populate the database — see server/.env.example for DB_* vars
npm run dev           # runs client (:5173) and server (:4000) together
```

**Deploy:**
- Client → Vercel. Framework preset auto-detects Vite; set the
  `VITE_API_BASE_URL` environment variable to the deployed API's `/api`
  base (e.g. `https://handmirror-server-production.up.railway.app/api`).
- Server → Railway (`npm start`). Needs `DB_HOST`, `DB_PORT`, `DB_USER`,
  `DB_PASSWORD`, `DB_NAME`, `CORS_ORIGIN` (the deployed client's origin), and
  `VIDEO_BASE_URL` — see `website/server/.env.example`.
