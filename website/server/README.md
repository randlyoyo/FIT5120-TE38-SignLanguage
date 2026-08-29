# Auslan website -- server

Node.js + Express + MySQL (via `mysql2`, plain SQL, no ORM). Serves the sign
library API consumed by `../client`.

## Setup

1. Get a MySQL 8 server running. Either use a local install, or Docker:
   ```
   docker run --name auslan-mysql -e MYSQL_ROOT_PASSWORD=yourpassword -p 3306:3306 -d mysql:8
   ```
2. Copy `.env.example` to `.env` and fill in your DB credentials.
3. Create the schema:
   ```
   mysql -u root -p < schema.sql
   ```
4. Install dependencies and seed placeholder data:
   ```
   npm install
   npm run seed
   ```
5. Run the dev server:
   ```
   npm run dev
   ```
   Listens on `http://localhost:4000` by default.

## For the data-science teammate: the `signs` table

See `schema.sql` for the full definition with column comments. The short
version:

| Column | Meaning |
| --- | --- |
| `gloss` | The sign's label/word, shown as the card title. Must be unique. |
| `definitions` | JSON array grouped by part of speech, Auslan-Signbank style: `[{"partOfSpeech":"Noun","senses":["...","..."]}]`. Placeholder rows use a single neutral `"Meaning"` group instead of inventing grammatical senses for a made-up gesture. |
| `usage_notes` | JSON array of short step strings, e.g. `["Step 1: ...", "Step 2: ..."]`. `mysql2` returns this already parsed as a JS array -- no manual `JSON.parse` needed. |
| `tags` | JSON array of **classification** labels, e.g. `["family","greeting"]`. A sign can carry more than one -- this replaces the old single-value `category` column, which couldn't represent a sign belonging to more than one group at once. Shown as visible chips in the UI, and used for browsing/filtering. |
| `keywords` | JSON array of **search-only** synonyms/alternate words, e.g. `["hi","hey"]` for a sign glossed `HELLO`. Never shown in the UI -- only used to match search queries. If you're unsure whether something is a `tag` or a `keyword`: is it a genuine classification/topic? -> `tags`. Is it just another word someone might type to find this sign? -> `keywords`. |
| `source` | Provenance note, e.g. `"Placeholder / Demo Data -- Not Real Auslan"` or `"Auslan Signbank (pending verification)"`. |
| `created_at` / `updated_at` | Set automatically by MySQL (`DEFAULT CURRENT_TIMESTAMP` / `ON UPDATE CURRENT_TIMESTAMP`). Never pass these in an INSERT/UPDATE -- just omit them. |

Example insert:
```sql
INSERT INTO signs (gloss, definitions, usage_notes, tags, keywords, source)
VALUES (
  'THANK YOU',
  CAST('[{"partOfSpeech":"Interjection","senses":["A polite expression of gratitude."]}]' AS JSON),
  CAST('["Touch fingertips to chin.", "Move hand forward toward the other person."]' AS JSON),
  CAST('["greeting"]' AS JSON),
  CAST('["thanks","cheers"]' AS JSON),
  'Auslan Signbank (pending verification)'
);
```

## API contract

Base URL: `http://localhost:4000/api` (proxied to `/api` by the Vite dev
server, so the frontend just calls `/api/...`).

### `GET /api/signs`

Query params (all optional): `query` (keyword search across `gloss`,
`tags`, `keywords` and `definitions`), `page` (default 1), `pageSize`
(default 6, max 50). There is no `category` param anymore -- classification
now lives in `tags`, which `query` already searches.

```
curl "http://localhost:4000/api/signs?query=hi&page=1"
```

```json
{
  "results": [
    {
      "id": 1,
      "gloss": "DEMO SIGN A",
      "definitions": [{ "partOfSpeech": "Meaning", "senses": ["Placeholder stand-in for a friendly hello-style greeting."] }],
      "usageNotes": ["Raise hand to shoulder height, palm facing forward.", "..."],
      "source": "Placeholder / Demo Data -- Not Real Auslan",
      "tags": ["greeting"],
      "keywords": ["hi", "hey", "wave hello"]
    }
  ],
  "pagination": { "page": 1, "pageSize": 4, "totalResults": 9, "totalPages": 3 },
  "query": { "query": "hi" }
}
```

### `GET /api/signs/:id`

```
curl "http://localhost:4000/api/signs/1"
```
Returns one object shaped like a single `results[]` entry above. `404` with
`{ "error": "Sign not found", "id": 42 }` if it doesn't exist.

### `POST /api/recognize` -- reserved stub, not implemented yet

This endpoint exists so whoever builds the AI recognition model has a fixed
contract to build against, without needing an API redesign later. Right now
it always returns `501 Not Implemented`:

```
curl -X POST "http://localhost:4000/api/recognize" -H "Content-Type: application/json" -d '{}'
```
```json
{
  "status": "not_implemented",
  "message": "Sign recognition is not implemented yet -- this endpoint reserves the contract for a future AI model.",
  "expectedRequestShape": {
    "targetSignId": "number",
    "landmarks": "Array<{timestamp:number, hand:\"left\"|\"right\", points:Array<{x:number,y:number,z:number}>}>"
  },
  "expectedResponseShapeWhenImplemented": {
    "status": "ok",
    "targetSignId": "number",
    "predictedLabel": "string",
    "confidence": "number (0-1)",
    "isMatch": "boolean"
  }
}
```

Landmarks are expected to be pre-extracted client-side (in-browser
MediaPipe, the same approach as the `web/` prototype) -- no video needs to
reach the server for this to work.
