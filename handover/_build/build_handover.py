"""Rebuild the four Handover Package documents.

Each document is rewritten in place on top of its own previous version, which
supplies the page setup, styles and footer; the helpers below reproduce the
original formatting (accent headings, navy table headers, red TEAM TO COMPLETE
boxes). Edit the content functions and re-run to produce the next version.

    python build_handover.py <handover dir>     (needs python-docx)

The previous version is archived by hand under handover/archive/vX.Y/ first.
"""
import copy
import sys
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

VERSION = "Version 2.1 · 13 October 2026 (living document -- updated each iteration) | Team TE38 · BitDance | FIT5120, Monash University"
ACCENT, INK, MUTED, BORDER = "1F3D7A", "201E19", "6B6455", "D9D3C4"
FLAG_INK, FLAG_FILL, FLAG_BORDER = "9C2F2F", "FBEFEF", "E3B8B8"
LIVE = "https://handmirror.vercel.app"
REPO = "https://github.com/randlyoyo/FIT5120-TE38-SignLanguage"
API = "https://fit5120-te38-signlanguage-production.up.railway.app"
GOVERNANCE = "https://drive.google.com/drive/folders/1u6BIRZyPxSSnvfm3VgOKKIQmPg2BWyyp"
MENTORS = "Afsana and Ted"
TEAM = [  # name, role, email, LinkedIn -- from the team's Team Info document (iteration 3)
    ["Dingao Wang", "Lead developer; support contact", "dwan0104@student.monash.edu", "www.linkedin.com/in/dingaowang"],
    ["Ziyang Wang", "Team member", "zwan0637@student.monash.edu", "www.linkedin.com/in/ziyang-wang-2b4700197"],
    ["Minshu Ye", "Team member", "myee0023@student.monash.edu", "www.linkedin.com/in/minshu-ye-24887b428"],
    ["Xinyue Mao", "Team member", "xmao0020@student.monash.edu", "www.linkedin.com/in/xinyue-mao-236877428"],
    ["Yan Zhuang", "Team member", "yzhu0233@student.monash.edu", "www.linkedin.com/in/yan-zhuang-b4b80b3a3"],
    ["Ruizi Song", "Team member; holds the Sign Chat model weights", "rson0034@student.monash.edu", "www.linkedin.com/in/ruizi-song-12a8a8392"],
]


class Doc:
    def __init__(self, path: Path):
        self.path = path
        self.d = Document(path)
        body = self.d.element.body
        for el in list(body.iterchildren()):
            if el.tag != qn("w:sectPr"):
                body.remove(el)

    # --- paragraphs ---
    def _p(self, before=None, after=None, style=None):
        p = self.d.add_paragraph(style=style)
        pf = p.paragraph_format
        if before is not None:
            pf.space_before = Pt(before / 20)
        if after is not None:
            pf.space_after = Pt(after / 20)
        return p

    @staticmethod
    def _run(p, text, color=INK, size=21, bold=False, italic=None, font=None):
        lines = text.split("\n")
        r = p.add_run(lines[0])
        for line in lines[1:]:
            r.add_break()
            r.add_text(line)
        r.font.color.rgb = RGBColor.from_string(color)
        r.font.size = Pt(size / 2)
        r.bold = bold or None
        if italic is not None:
            r.italic = italic
        if font:
            r.font.name = font
            r._r.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), font)
        return r

    def header(self, kicker, title="HandMirror"):
        self._run(self._p(after=40), kicker, ACCENT, 20, bold=True)
        self._run(self._p(after=40), title, INK, 48, bold=True)
        self._run(self._p(after=280), VERSION, MUTED, 20, italic=True)

    def h1(self, text):
        self._run(self._p(before=280, after=120), text, ACCENT, 28, bold=True)

    def h2(self, text):
        self._run(self._p(before=160, after=80), text, INK, 23, bold=True)

    def para(self, text):
        self._run(self._p(after=160), text, INK, 21, italic=False)

    def code(self, text):
        self._run(self._p(after=160), text, INK, 18, italic=False, font="Consolas")

    def bullets(self, items):
        for it in items:
            p = self._p(after=60, style="List Bullet")
            self._run(p, it, INK, 21)

    def closing(self, text):
        self._run(self._p(before=320), text, MUTED, 18)

    # --- tables ---
    @staticmethod
    def _borders(tbl, color):
        tblPr = tbl._tbl.tblPr
        b = OxmlElement("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), "4")
            e.set(qn("w:space"), "0")
            e.set(qn("w:color"), color)
            b.append(e)
        tblPr.append(b)

    @staticmethod
    def _cell(cell, width, fill, margins):
        tcPr = cell._tc.get_or_add_tcPr()
        w = OxmlElement("w:tcW")
        w.set(qn("w:type"), "dxa")
        w.set(qn("w:w"), str(width))
        tcPr.append(w)
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), fill)
        tcPr.append(shd)
        mar = OxmlElement("w:tcMar")
        for side, v in zip(("top", "bottom", "start", "end"), margins):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:w"), str(v))
            e.set(qn("w:type"), "dxa")
            mar.append(e)
        tcPr.append(mar)

    def table(self, headers, rows, widths):
        t = self.d.add_table(rows=1 + len(rows), cols=len(headers))
        t.alignment = 1  # centre
        self._borders(t, BORDER)
        for r, values in enumerate([headers] + rows):
            for c, value in enumerate(values):
                cell = t.cell(r, c)
                self._cell(cell, widths[c], ACCENT if r == 0 else "FFFFFF", (100, 100, 140, 140))
                p = cell.paragraphs[0]
                self._run(p, value, "FFFFFF" if r == 0 else INK, 19, bold=(r == 0))
        self._p(after=80)

    def flag(self, text):
        t = self.d.add_table(rows=1, cols=1)
        self._borders(t, FLAG_BORDER)
        cell = t.cell(0, 0)
        self._cell(cell, 9972, FLAG_FILL, (80, 80, 140, 140))
        p = cell.paragraphs[0]
        self._run(p, "TEAM TO COMPLETE: ", FLAG_INK, 19, bold=True)
        self._run(p, text, FLAG_INK, 19)
        self._p(after=80)

    def save(self):
        self.d.save(self.path)


# ---------------------------------------------------------------- content ---

SHARED_LICENCE_ROWS = [
    ["Sign records & demonstration videos", "MM-WLAuslan word-level dataset (UQ CV Lab); dictionary cross-reference to Auslan Signbank", "Research dataset terms; Signbank is CC BY-NC-ND 4.0"],
    ["Practice recognition model", "Trained by the team on MM-WLAuslan", "Inherits the dataset's terms"],
    ["3D avatar body", "SMPL-X neutral model (Max Planck Institute for Intelligent Systems)", "Non-commercial research licence; the model may not be redistributed"],
    ["Avatar motion", "SMPL-X parameters fitted by the team to the MM-WLAuslan signers", "Inherits the dataset's terms"],
    ["Avatar skin", "BEDLAM body textures by Meshcapade", "CC BY-NC 4.0"],
    ["Avatar clothing", "Meshcapade SMPL texture samples", "CC BY 4.0"],
    ["Avatar hair", "Exported from an Avaturn avatar", "Not yet confirmed"],
    ["Scenario pictograms", "Human Pictogram 2.0 by TopeconHeroes (recoloured)", "Free to use as part of a design; files may not be redistributed or sold on their own"],
    ["Sign Chat models", "Uni-Sign (sign-to-text) and SignSparK (text-to-sign), fine-tuned on Auslan-Daily; Qwen2.5-1.5B-Instruct for dialogue", "Per each upstream project's licence"],
    ["Home page cover footage", "Four clips of people signing from Pexels, video IDs 10373883, 10374296, 10374298, 10373841 (client/public/hero)", "Pexels licence: free to use, attribution not required"],
]


def product(d: Doc):
    d.header("HANDOVER PACKAGE — PRODUCT DOCUMENT")
    d.h1("1. Overview")
    d.para(
        "HandMirror is a free, web-based Auslan (Australian Sign Language) learning assistant for families. "
        "This version delivers the full learning loop the team planned: look up a sign; learn it three ways "
        "(a real signer on video, a step-by-step breakdown, and a 3D avatar you can turn to any angle); practise "
        "it on camera with instant feedback; organise learning with personalised To Learn lists and real-life "
        "scenario journeys; and hold a short conversation with a signing avatar (Sign Chat)."
    )
    d.h2("The problem")
    d.para(
        "Parents and caregivers of young Deaf or hard-of-hearing children in Australia often have limited Auslan "
        "knowledge and struggle to find practical, trustworthy learning support for everyday family communication. "
        "Australian news coverage (ABC News, 2022) documented a family that secured NDIS funding but still could not "
        "find a suitable Auslan tutor -- funding does not always create access. Existing online resources are "
        "scattered, often move too fast for a beginner, and cannot confirm whether a learner is practising a sign "
        "correctly."
    )
    d.h2("The audience")
    d.para(
        "Primary audience: parents and caregivers of young Deaf or hard-of-hearing children in Australia, with "
        "limited Auslan knowledge and a need for flexible, family-friendly support. Primary persona: “Sarah "
        "Thompson”, 38, a parent and primary-school teacher who needs short, flexible learning sessions with "
        "progress tracking. Secondary audiences: Auslan beginners generally, and accessibility-focused learners "
        "(students, teachers)."
    )
    d.h2("Why it's important")
    d.para(
        "The project addresses UN Sustainable Development Goal 4 (Quality Education) and Goal 10 (Reduced "
        "Inequalities) -- giving families a flexible, low-cost way to build practical Auslan at home, rather than "
        "depending on scarce tutor availability."
    )

    d.h1("2. What's built so far")
    d.table(
        ["Epic", "Status", "What it does today"],
        [
            ["Epic 1 -- Sign library search & browse", "Delivered",
             "Keyword search across gloss, tags, keywords and definitions; topic browsing; difficulty-level filter and sort; pagination; "
             "sign-to-search (sign at the webcam and pick from the four closest signs)."],
            ["Epic 2 -- Sign demonstration & learning", "Delivered",
             "Real signer video for every sign (0.5×–2× speed), a step-by-step movement description, a rotatable 3D avatar "
             "playing the sign, and Mark-as-learned."],
            ["Epic 3 -- Basic quiz / self-check", "Not built", "Remains open for a future iteration."],
            ["Epic 4 -- AI sign recognition & webcam feedback", "Delivered",
             "Practice tab: the learner signs at the webcam and gets one of three results -- Excellent (score 80+), "
             "Almost there (60–79) or Keep practicing (under 60) -- with a tip. Either hand may lead."],
            ["Epic 5 -- Personalised learning", "Delivered",
             "To Learn list (grouped by when signs were added), a personalise wizard (pick topics, a word count and its split; "
             "it fills the list with signs not yet learned), and a Learned list with progress."],
            ["Epic 6 -- Family mode", "Not built", "Remains open for a future iteration."],
            ["Epic 7 -- Real-life conversation practice", "Delivered",
             "Scenario Learning: 34 journeys (e.g. café, doctor, family) split into short stops, with a guided route and progress. "
             "Sign Chat: type or sign a sentence and a 3D avatar replies in Auslan."],
        ],
        [2600, 1300, 6072],
    )

    d.h1("3. How it works (high level)")
    d.para(
        "A React + TypeScript client (hosted on Vercel) calls a Node.js + Express REST API (hosted on Railway), backed "
        "by a MySQL database (also on Railway). Demonstration videos are served from Cloudflare R2. Practice runs the "
        "hand-tracking in the learner's browser and scores the attempt with a small recognition model inside the "
        "Express API. Sign Chat uses a separate Python model service that needs a GPU. Full technical detail is in "
        "the Support and Maintenance Documents."
    )
    d.para(f"Live product: {LIVE}   |   Source code: {REPO}")

    d.h1("4. Ethics & responsible design")
    d.h2("Data provenance and licensing")
    d.para(
        "Every sign record carries a source field recording where its content came from, so the product never "
        "silently presents unverified content as confirmed Auslan. The third-party assets in the product, and the "
        "terms they come under, are:"
    )
    d.table(["Asset", "Source", "Licence / terms"], SHARED_LICENCE_ROWS, [2400, 4300, 3272])
    d.flag(
        "several assets above are licensed for non-commercial use only (SMPL-X, the BEDLAM skin texture, Auslan "
        "Signbank, the research datasets). Confirm the terms of each against how a sponsor intends to use the product "
        "before any commercial use, and confirm whether the Avaturn hair may be used on a public website."
    )
    d.h2("Content safety")
    d.para(
        "The dictionary covers the full Auslan vocabulary of its source dataset, which includes adult and crude signs -- "
        "appropriate for a reference, but not for a family homepage. The home page's sign strip therefore draws only "
        "from a fixed list of 16 family-friendly signs (HELLO, THANK YOU, DOG, MOTHER and so on), never from a random "
        "catalogue pick. The cover footage behind the home page title is stock video of people signing, chosen for "
        "mood; it is not presented as a lesson."
    )
    d.h2("Privacy")
    d.bullets([
        "No user accounts. Learned and To Learn lists are stored only in the visitor's own browser (localStorage) and are never sent to the server.",
        "Practice: camera video never leaves the browser. Hand-tracking runs locally; only numeric landmark coordinates are posted for scoring, and they are not stored.",
        "Sign Chat: typed messages are sent to the Sign Chat service. When a learner chooses to sign a message instead, the recorded clip is uploaded to that service so it can be recognised.",
    ])
    d.para("What the Sign Chat service keeps (checked in sign_chat_backend/signchat, 13 October 2026):")
    d.bullets([
        "Uploaded video clips: written to a temporary file for recognition and deleted as soon as the request finishes, whether it succeeds or fails.",
        "Conversation text: the last 6 exchanges of each session, in memory only (at most 1,000 sessions). Lost when the service restarts; DELETE /api/session/:id clears one sooner.",
        "Generated replies: each reply's motion file, skeleton video and subtitles (which contain the reply text) are saved under media/ and are not cleaned up automatically. On Colab they disappear when the runtime ends; a permanent host should delete them after a set period.",
        "No accounts, names or IP addresses are recorded by the service itself.",
    ])
    d.para("Recommendation: tell users on the Sign Chat page that their messages are processed by the service and not kept beyond the session.")
    d.h2("Honesty about limits")
    d.para(
        "The interface tells learners that the 3D avatar shows the overall movement and that exact handshapes should be "
        "taken from the demonstration video. The Practice score threshold was calibrated on studio footage, and Sign "
        "Chat's replies have not yet been checked by fluent signers -- see the Support Document's known limitations."
    )
    d.h2("ACS Code of Conduct mapping")
    d.table(
        ["ACS value", "How HandMirror addresses it"],
        [
            ["The public interest", "Targets a documented, real accessibility gap for Deaf/hard-of-hearing families rather than a purely commercial idea."],
            ["Quality of life", "Free to use, no account required, reduces dependence on scarce and costly in-person tutoring."],
            ["Honesty", "Source fields disclose provenance; the avatar is labelled as a guide; adult signs are kept off the family-facing home page."],
            ["Competence", "Recognition has a parity test against the training pipeline; the APIs and data paths are documented for safe extension."],
            ["Professional development", "The recognition contract (recognition/API.md) and Sign Chat README let future contributors retrain or replace models without redesigning the product."],
            ["Professionalism", "Known limitations and unconfirmed licences are disclosed rather than hidden."],
        ],
        [2400, 7572],
    )
    d.para("The full Ethics Canvas is in the Project Governance Portfolio: " + GOVERNANCE)

    d.h1("5. Team")
    d.table(["Name", "Role", "Email", "LinkedIn"], TEAM, [2000, 2400, 2700, 2872])
    d.para("Team TE38 · “BitDance” · FIT5120, Monash University · Studio mentors: " + MENTORS + ".")

    d.h1("6. Potential sponsors")
    d.table(
        ["Organisation", "Why they fit"],
        [
            ["Deaf Children Australia", "Runs family programs and Auslan resources; could validate family needs and test the platform directly with parents and caregivers."],
            ["Expression Australia", "Provides Auslan education and services (incl. Auslan in the Home, Auslan Storytime); could advise on Auslan accuracy, accessibility and culturally appropriate design."],
        ],
        [2600, 7372],
    )

    d.h1("7. Links")
    d.table(
        ["Item", "Link"],
        [
            ["Live product", LIVE],
            ["Source code (GitHub)", REPO],
            ["System Architecture Document", "website/docs/system-architecture.docx (predates Practice, the 3D avatar and Sign Chat -- see the Maintenance Document)"],
            ["Product demo video", "Project Governance Portfolio → Project Support folder"],
            ["Team LinkedIn", "Each member's profile is listed in Section 5"],
            ["Project Governance Portfolio (Google Drive)", GOVERNANCE],
        ],
        [3400, 6572],
    )
    d.closing("HandMirror — Product Document · Team TE38 (BitDance) · FIT5120")


def support(d: Doc):
    d.header("HANDOVER PACKAGE — SUPPORT DOCUMENT")
    d.h1("1. Purpose & audience")
    d.para(
        "This document is for whoever operationally runs and supports the live HandMirror product once a sponsor has "
        "taken it on -- day-to-day operation, data loading, backups, and what to do when something breaks. It assumes "
        "no prior familiarity with the codebase (that's the companion Maintenance Document)."
    )
    d.h1("2. System summary")
    d.table(
        ["Component", "Technology", "Where it runs"],
        [
            ["Web client", "React 19 + TypeScript (Vite); three.js for the 3D avatar; MediaPipe for in-browser hand-tracking", "Vercel"],
            ["API", "Node.js + Express; ONNX Runtime for the Practice recognition model", "Railway"],
            ["Database", "MySQL (signs, sign_videos, sign_poses)", "Railway"],
            ["Demonstration videos", "MP4 files", "Cloudflare R2"],
            ["Sign Chat service", "Python + FastAPI; sign-to-text, text-to-sign and dialogue models; needs a GPU", "See Section 3"],
        ],
        [2300, 5000, 2672],
    )
    d.h1("3. Environments & access")
    d.table(
        ["Environment", "Component", "Address"],
        [
            ["Production", "Client (Vercel)", LIVE],
            ["Production", "API (Railway)", API],
            ["Production", "Database", "MySQL on Railway (shared with the API service)"],
            ["Production", "Sign Chat service", "Google Colab GPU runtime behind a temporary trycloudflare.com URL (below)"],
            ["Preview", "Client", "Vercel builds a preview for every non-master branch (e.g. staging)"],
            ["Local dev", "Client + API", "cd website && npm run install:all && npm run seed && npm run dev (client :5173, API :4000)"],
            ["Local dev", "Sign Chat (no GPU)", "cd sign_chat/sign_chat_backend && SIGNCHAT_MOCK=1 SIGNCHAT_DIALOGUE=echo uvicorn signchat.server:app --port 8000"],
        ],
        [1700, 2300, 5972],
    )
    d.h2("Sign Chat hosting")
    d.para(
        "Sign Chat has no permanent host yet. It runs from sign_chat/sign_chat_backend/colab_server.ipynb on a Colab "
        "A100 (an L4 with High-RAM also works; peak GPU memory is about 15 GB). Run sections 1-5 only: section 6 holds "
        "optional checks that stop and restart the server, which leaves the site without Sign Chat for about two "
        "minutes. Section 5 prints the public URL, which changes every time the runtime restarts."
    )
    d.bullets([
        "Model weights (about 13 GB) are in the Google Drive of ruizisong11@gmail.com, folders auslan_work/ and smplx_models/. The notebook mounts the Drive of the account running it, so either run it from that account or share both folders and add them as shortcuts at the top of My Drive. They are not in the repository and cannot be downloaded from anywhere else -- copy them to storage the sponsor owns at handover.",
        "The production client reads the URL from website/client/.env.production. When the URL changes, edit that line and push to master (develop → staging → master); Vercel rebuilds automatically. Local development uses website/client/.env.",
        "While the runtime is up, anyone with the URL can call the service. Stop the runtime when it is not needed.",
    ])
    d.h2("Deployment triggers")
    d.para(
        "Both platforms deploy from the master branch automatically: Vercel builds the client, and Railway (project "
        "“pure-kindness”, service at " + API + ") redeploys the API. Pushes to other branches produce Vercel previews only."
    )
    d.h1("4. Running & operating the product")
    d.para(
        "Client: a static Vite build, auto-deployed by Vercel on every push to the master branch (code flows develop → "
        "staging → master). API: a long-running Node process on Railway, started with npm start and restarted "
        "automatically on crash; GET /api/health returns {\"status\":\"ok\"} when it is up. The Practice model files "
        "(recognition/models, about 14 MB) are committed to the repository, so nothing needs fetching at deploy time. "
        "The 3D avatar (client/public/smplx.glb, 6.2 MB) is a static file deployed with the client."
    )
    d.h2("Required environment variables")
    d.table(
        ["Variable", "Where", "Purpose"],
        [
            ["PORT", "Server", "Port the Express server listens on (default 4000)."],
            ["DB_HOST / DB_PORT", "Server", "MySQL host and port."],
            ["DB_USER / DB_PASSWORD", "Server", "MySQL credentials -- never commit real values."],
            ["DB_NAME", "Server", "Database name (auslan_learning)."],
            ["CORS_ORIGIN", "Server", "The exact origin allowed to call the API. If unset it defaults to “*” (any origin) -- always set it in production."],
            ["VIDEO_BASE_URL", "Server", "Base URL used to resolve a sign's demonstration video when its row has no explicit video_url."],
            ["RECOGNITION_MODELS", "Server", "Optional path to the recognition model folder; defaults to the repository's recognition/models."],
            ["VITE_API_BASE_URL", "Client", "Base URL the client calls for the API, including /api (e.g. " + API + "/api)."],
            ["VITE_SIGNCHAT_API_BASE_URL", "Client", "Base URL of the Sign Chat service. Production: committed in website/client/.env.production (Section 3)."],
        ],
        [2700, 1100, 6172],
    )
    d.para(
        "Secret values live only in each platform's project settings (Vercel / Railway environment variables), never in "
        "source control -- see Section 6. The one exception is the Sign Chat URL, which is public anyway (it is built "
        "into the client) and is committed so that changing it is a one-line push."
    )

    d.h1("5. Data management")
    d.para(
        "The database has three tables: signs (one row per sign -- gloss, definitions, usage notes, tags, keywords and a "
        "source provenance string, several columns stored as native JSON), sign_videos (one or more demonstration videos "
        "per sign) and sign_poses (one gzipped avatar motion clip per sign, keyed by sign id; a sign without a row simply "
        "has no 3D avatar)."
    )
    d.h2("Loading data")
    d.bullets([
        "npm run seed -- loads a small set of clearly-labelled placeholder demo signs (local development only).",
        "npm run import:real -- imports the real dataset from server/data/signs.json and server/data/sign_videos.json.",
        "Avatar motion: python scripts/convert_poses.py <npz dir> <out dir> turns fitted SMPL-X .npz files into .json.gz, then node scripts/importPoses.js <out dir> upserts them into sign_poses, matched by gloss (run from website/server; it writes to whichever database the server's .env points at).",
    ])
    d.h2("Backup & restore")
    d.para(
        "Railway backs up the production MySQL volume automatically. To restore, open the MySQL service in the Railway "
        "dashboard, go to its Backups tab, pick a backup and choose Restore; the API reconnects on its own once the "
        "database is back. For a copy outside Railway (e.g. before a risky data import), take a manual dump:"
    )
    d.code("mysqldump -h <DB_HOST> -P <DB_PORT> -u <DB_USER> -p auslan_learning > backup.sql\nmysql -h <DB_HOST> -P <DB_PORT> -u <DB_USER> -p auslan_learning < backup.sql")

    d.h1("6. Security & confidential information")
    d.para(
        "There are no end-user accounts and no authentication layer. The sign endpoints are read-only; the only POST "
        "endpoints score a Practice attempt or look a sign up from landmark data, and none of them write to the database. "
        "Database credentials and other secrets exist only as environment variables in the Vercel/Railway project "
        "settings; the local .env file is git-ignored (.env.example documents the keys with no real values)."
    )
    d.h2("Known limitations")
    d.bullets([
        "No rate-limiting middleware. This matters more than before: each POST /api/recognize call runs model inference on the API server's CPU, so a flood of requests could slow the whole API.",
        "No helmet or equivalent security-headers middleware.",
        "Request bodies up to 5 MB are accepted (a Practice capture is about 190 KB).",
        "CORS falls back to allowing any origin if CORS_ORIGIN is not set.",
        "The Practice score threshold was calibrated on studio (green-screen) footage; home webcams, lighting and framing can lower scores.",
        "Sign Chat's sign-to-text and text-to-sign models have not been validated by fluent signers; they do best on short, everyday sentences.",
    ])
    d.para("Recommendation before wider public or commercial use: add express-rate-limit (with a tighter limit on /api/recognize) and helmet to the Express server, and always set CORS_ORIGIN.")

    d.h1("7. API reference summary")
    d.table(
        ["Endpoint", "Notes"],
        [
            ["GET /api/health", "Liveness check; returns {\"status\":\"ok\"}."],
            ["GET /api/signs", "Search/list signs. Optional: query (search across gloss, tags, keywords, definitions), tag, level (beginner / intermediate / advanced), sort, page, pageSize (default 6, max 50)."],
            ["GET /api/signs/tags", "Topic categories with counts. Optional exclude (comma-separated sign ids) so counts reflect what is still left to learn."],
            ["GET /api/signs/sample", "Random signs from one topic for the personalise wizard: tag, count (max 50), exclude."],
            ["GET /api/signs/:id", "Single sign, including its demonstration videos. 404 if the id doesn't exist."],
            ["GET /api/signs/:id/pose", "The sign's avatar motion clip (SMPL-X joint rotations per frame). 404 if the sign has none."],
            ["POST /api/recognize/verify", "Practice scoring. Body {word, capture}; returns matched, a 0–100 score, a tier (excellent / close / needs_practice) and a feedback sentence."],
            ["POST /api/recognize/identify", "Sign-to-search. Body {capture}; returns the four closest signs with a calibrated confidence."],
            ["GET /api/recognize/vocabulary", "Which signs the recognition model knows (?detail=1 adds the Signbank cross-reference)."],
            ["Sign Chat service", "POST /api/chat/text, POST /api/chat/sign (multipart video upload), DELETE /api/session/:id -- see sign_chat/sign_chat_backend/README.md."],
        ],
        [3000, 6972],
    )
    d.para("The landmark “capture” format and error codes (400 unusable_capture, 404 unknown_word, 503 model_unavailable) are specified in recognition/API.md.")

    d.h1("8. Troubleshooting guide")
    d.table(
        ["Symptom", "Likely cause / fix"],
        [
            ["Server won't start", "Check DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME point to a reachable MySQL instance."],
            ["CORS error in the browser console", "CORS_ORIGIN on the server doesn't match the client's actual deployed origin."],
            ["Videos don't play / show black", "VIDEO_BASE_URL isn't set, or the Cloudflare R2 bucket it points to is unreachable."],
            ["Search returns zero results", "The database hasn't been seeded or imported yet -- run npm run seed or npm run import:real."],
            ["Practice says “isn't switched on for this site yet”", "The API returned 503 model_unavailable: recognition/models is missing from the deployment, or RECOGNITION_MODELS points at the wrong folder."],
            ["Practice says “Didn't catch enough movement”", "400 unusable_capture: both hands weren't in frame, or the capture was too short. Ask the learner to step back so both shoulders are visible."],
            ["“There's no 3D avatar for this sign yet”", "The sign has no sign_poses row -- import its motion clip (Section 5)."],
            ["3D avatar panel stays blank", "smplx.glb failed to load, or the browser has WebGL disabled; check the browser console."],
            ["Sign Chat replies fail", "Open <URL>/api/health. No answer: the Colab runtime has stopped or its URL changed -- rerun sections 1-5 and update website/client/.env.production. A 502 for a minute or two right after a restart is normal while the models load."],
            ["Sign Chat repeats your text back and the avatar barely moves", "The client is talking to a mock server (SIGNCHAT_MOCK=1, echo dialogue), not the real models -- check which URL VITE_SIGNCHAT_API_BASE_URL points at."],
        ],
        [3300, 6672],
    )
    d.h1("9. Support contacts")
    d.table(
        ["Name", "Role", "Contact"],
        [
            ["Dingao Wang", "Lead developer -- first contact for any issue", "wangdingao5337@gmail.com"],
            ["Ruizi Song", "Holds the Sign Chat model weights (Google Drive)", "ruizisong11@gmail.com"],
        ],
        [2400, 4300, 3272],
    )
    d.closing("HandMirror — Support Document · Team TE38 (BitDance) · FIT5120")


def maintenance(d: Doc):
    d.header("HANDOVER PACKAGE — MAINTENANCE DOCUMENT")
    d.h1("1. Purpose")
    d.para(
        "This document is for a future technical team extending HandMirror -- adding features, fixing bugs, retraining "
        "models, or building out the epics still on the roadmap. It assumes the reader is a competent developer new to "
        "this specific codebase."
    )
    d.h1("2. Codebase map")
    d.table(
        ["Path", "What's there"],
        [
            ["website/client/src/pages", "One component per route: Home, SignLibrary, SignDetail, Scenario list and journey (ScenarioCategoryPage), ToLearn, LearnedSigns, Conversation (Sign Chat)."],
            ["website/client/src/components", "Shared UI. Key pieces: SmplxAvatar (3D avatar), SignDemonstration (video / 3D avatar tabs), PracticeVerify and SignCaptureModal (webcam capture), GestureSearch (sign-to-search), PersonalizeSessionPanel + PieRatioChart (personalise wizard), SavedSignsPage + TimeGroupedList (To Learn / Learned)."],
            ["website/client/src/api", "Typed fetch wrappers: signs.ts (dictionary), recognize.ts (Practice / sign-to-search), signChat.ts (Sign Chat service)."],
            ["website/client/src/lib", "landmarks.ts (MediaPipe capture with motion-based auto-stop), smplx.ts (avatar pose format), scenarioCategories.ts (the 34 journeys and their words), pictograms.ts (journey icons), learnedSigns / toLearnSigns / idListStore (localStorage lists)."],
            ["website/client/src/assets/pictograms", "The 36 journey pictograms (Human Pictogram 2.0, recoloured)."],
            ["website/client/public/smplx.glb", "The 3D avatar: SMPL-X body with texture, hair and a “smile” morph target."],
            ["website/client/src/styles/theme.css", "The whole design system in one file -- colour/type/radius tokens plus every page's styles."],
            ["website/server/src/routes", "signs.js (dictionary and avatar poses), recognize.js (Practice and sign-to-search)."],
            ["website/server/src/recognition", "features.js (landmarks → model input, a port of the Python training pipeline) and model.js (ONNX encoder and template bank)."],
            ["website/server/src/utils", "formatSign / pagination / video helpers, difficulty.js (level heuristic), popularity.js (popularity sort)."],
            ["website/server/scripts", "importRealData.js, convert_poses.py and importPoses.js (data loading -- see the Support Document)."],
            ["recognition/", "Model files, the interface specification (API.md), deployment notes (ASSETS.md), build scripts and the parity fixtures (test/golden.json)."],
            ["sign_chat/sign_chat_backend", "The Sign Chat model service (FastAPI), its README, Colab launcher and tests."],
        ],
        [3100, 6872],
    )

    d.h1("3. Adding or editing content")
    d.h2("Signs")
    d.para("gloss has a UNIQUE constraint, so an insert can double as an upsert:")
    d.code(
        "INSERT INTO signs (gloss, definitions, usage_notes, tags, keywords, source)\n"
        "VALUES ('THANK YOU',\n"
        "  CAST('[{\"partOfSpeech\":\"Interjection\",\"senses\":[\"A polite expression of gratitude.\"]}]' AS JSON),\n"
        "  CAST('[\"Touch fingertips to chin.\", \"Move hand forward toward the other person.\"]' AS JSON),\n"
        "  CAST('[\"greeting\"]' AS JSON),\n"
        "  CAST('[\"thanks\",\"cheers\"]' AS JSON),\n"
        "  'Auslan Signbank (pending verification)');"
    )
    d.para(
        "tags are visible classification chips (a sign can have several); keywords are search-only synonyms that are "
        "never shown in the UI. When in doubt: a genuine topic/category is a tag, another word someone might type to "
        "find the sign is a keyword."
    )
    d.h2("Scenario journeys")
    d.para(
        "Journeys and their stops live in client/src/lib/scenarioCategories.ts (words are matched to signs by gloss). "
        "A new journey also needs an entry in client/src/lib/pictograms.ts mapping its id to a pictogram file."
    )
    d.h2("Avatar motion")
    d.para("Add or replace a sign's 3D motion with convert_poses.py and importPoses.js (Support Document, Section 5). The clip must be fitted to the same SMPL-X body the avatar uses (Section 6).")

    d.h1("4. Extending the API")
    d.bullets([
        "One express.Router per resource, mounted in src/index.js -- follow the existing pattern in src/routes/signs.js for a new resource.",
        "Always use parameterized ? placeholders via mysql2 -- never string-concatenate user input into SQL.",
        "Map DB rows to the API's camelCase JSON shape through a small helper in src/utils/ (see formatSign.js), keeping snake_case DB columns out of the public response shape.",
    ])

    d.h1("5. Practice recognition")
    d.para(
        "MediaPipe Holistic runs in the browser and records 48 landmarks per frame (arms and both hands); the capture "
        "stops itself shortly after the hands come to rest. The API turns the landmarks into a fixed 96 × 98 tensor "
        "(features.js), runs the encoder (encoder.onnx, 256-dimensional output) and compares the result with the "
        "target sign's 12 stored templates by cosine distance."
    )
    d.bullets([
        "Verify (Practice): matched when the distance is below τ = 0.529 -- the equal-error point on the validation split (EER 0.73%). The 0–100 score is a rescaling of that distance anchored so 60 is exactly the pass line.",
        "Left- or right-handed: the attempt is scored as captured and mirrored (other hand leading), and the closer result is kept. This slightly raises the chance of accepting a wrong sign.",
        "Identify (sign-to-search): distance to every sign; the four closest are returned with a confidence calibrated on the validation split.",
        "Any change to features.js must keep npm run test:recognition (in website/server) passing -- it checks the JavaScript pipeline against fixtures exported from the Python original.",
        "After retraining, rebuild the template bank with recognition/scripts/buildBank.mjs and follow recognition/ASSETS.md. The full contract, including capture guidance for the UI, is recognition/API.md.",
    ])

    d.h1("6. The 3D avatar")
    d.para(
        "SmplxAvatar loads /smplx.glb: the SMPL-X neutral body with one bone per SMPL-X joint, named after the joint, "
        "with identity rest rotations. Each frame's SMPL-X axis-angle rotations are applied straight onto those bones -- "
        "no retargeting -- so the avatar reproduces the fitted motion, including where the hands touch. On top of the "
        "motion the component adds eyes that follow the camera within ±8°, a resting smile on Sign Chat only (a morph "
        "target), and an idle pose set to the median trunk and head angle of the signing clips."
    )
    d.para(
        "Pose clips are JSON: {fps, frames, smplx: {global_orient, body_pose (21 joints × 3), left_hand_pose and "
        "right_hand_pose (15 × 3 each)}}, stored gzipped in sign_poses. They come from SMPL-X fits of the dataset's "
        "signers; motion fitted to any other body would not line up."
    )
    d.para(
        "The scripts that fit SMPL-X to the dataset and build smplx.glb (body, textures, clothes, hair) are in the "
        "Project Governance Portfolio as HandMirror-avatar-build-scripts.zip, with a README listing the order to run "
        "them and the input files. The SMPL-X model, BEDLAM skin and Meshcapade textures are not in the zip, because "
        "their licences forbid redistribution; download them from their official sites (Product Document, Section 4)."
    )

    d.h1("7. Sign Chat service")
    d.para(
        "A separate Python service (sign_chat/sign_chat_backend). A typed message goes to a dialogue model; a signed "
        "message is first turned into text by a sign-to-text model; the reply text then goes through a text-to-sign "
        "model that produces the avatar motion the client plays. Its README covers the models, the API, a mock mode for "
        "front-end work without a GPU (SIGNCHAT_MOCK=1), and the tests (python -m pytest -q tests)."
    )

    d.h1("8. Coding conventions observed")
    d.bullets([
        "Client: TypeScript in strict mode (React 19 + Vite); lint with npm run lint (oxlint).",
        "Server: plain CommonJS Node.js (no TypeScript, no ORM -- raw parameterized SQL via mysql2).",
        "camelCase at the API/JSON boundary; snake_case for DB columns -- translated by the format helpers, never leaked across that boundary in either direction.",
        "Design system lives entirely in client/src/styles/theme.css: IBM Plex Mono for display/labels, IBM Plex Sans for body text, with colour/spacing/radius as CSS custom properties.",
        "Comments explain why a non-obvious choice was made; keep that habit when changing the recognition or avatar code, where the reasons are not visible in the code itself.",
    ])

    d.h1("9. Standards & compliance")
    d.h2("Data and asset licensing")
    d.para(
        "Every sign record's source field must stay accurate. The licences of all third-party assets are listed in the "
        "Product Document, Section 4 -- several are non-commercial, and the SMPL-X model may not be redistributed."
    )
    d.flag("confirm each licence against the sponsor's intended use before any commercial use, including the Avaturn hair.")
    d.h2("Accessibility")
    d.para(
        "Conformance level: WCAG 2.1 Level AA. Checked on 13 October 2026 with axe-core 4.10.2 (all WCAG 2.0, 2.1 and "
        "2.2 A/AA rules) in Chrome on every page: home, library, sign detail, scenarios, scenario categories, To Learn, "
        "Learned and Sign Chat. The demonstration videos have no audio, so captions are not required."
    )
    d.para(
        "Keep it at AA when changing the interface: text needs a contrast of at least 4.5:1 (3:1 for large text), "
        "every control and progress bar needs an accessible name, and re-run axe on any page you change."
    )
    d.h2("Security hygiene going forward")
    d.bullets([
        "Secrets stay in environment variables only -- never hard-coded or committed.",
        "Add rate limiting (stricter on /api/recognize) and helmet before wider use; set CORS_ORIGIN explicitly.",
        "Any future write-capable endpoint must add real authentication/authorisation.",
        "Run npm audit periodically on both client and server and address high-severity advisories.",
    ])

    d.h1("10. Roadmap / open epics")
    d.table(
        ["Item", "Status"],
        [
            ["Basic quiz / self-check (Epic 3)", "Not yet built."],
            ["Family mode for parent-child signing (Epic 6)", "Not yet built."],
            ["Permanent hosting for the Sign Chat service", "Runs from a Colab notebook today, with weights in a team member's Google Drive (Support Document, Section 3). Needs a GPU host with at least 24 GB (e.g. an L4) and a fixed URL."],
            ["Facial expression on the avatar", "Auslan grammar uses the face (e.g. raised brows for questions); the avatar's face is not driven yet."],
            ["Validation by fluent signers", "Practice thresholds and Sign Chat output have not yet been checked by Auslan signers."],
        ],
        [4300, 5672],
    )

    d.h1("11. Known technical debt")
    d.bullets([
        "ScenarioLearningPage (route /scenarios/:id) and its story data (scenarioStories.ts, useScenarioProgress, ScenarioWordLink) are no longer linked from anywhere and can be removed.",
        "Unused leftovers: the @pixiv/three-vrm dependency, client/public/avatar.vrm (untracked), and style rules in theme.css for the previous home page and scenario sidebar.",
        "smplx.glb is 6.2 MB, mostly the hair mesh; simplifying the hair would speed up the first avatar load.",
        "The home page plays up to 33 short videos at once (the cover plus the sign strip); consider fewer clips or playing them only on hover for low-end devices.",
        "The four cover clips in client/public/hero are the original 1080p Pexels downloads (2.7-9 MB each, 22 MB in total; one plays at a time). Re-encoding them to 720p with ffmpeg would cut them to about a quarter of that.",
        "Automated tests: the recognition parity test (server) and the Sign Chat service's pytest suite. There are no client tests and no CI pipeline beyond Vercel/Railway building on push.",
    ])
    d.closing("HandMirror — Maintenance Document · Team TE38 (BitDance) · FIT5120")


def checklist(d: Doc):
    d.header("HANDOVER PACKAGE", "Handover Checklist and Sign-Off")
    d.para(
        "We, Team TE38 (“BitDance”), acknowledge that we have provided all the handover documents listed below to the "
        "studio mentors named below, via the Project Governance Portfolio (Google Drive link below)."
    )
    d.table(
        ["Item", "Details"],
        [
            ["Team members", ", ".join(m[0] for m in TEAM)],
            ["Studio mentors", MENTORS],
            ["Project Governance Portfolio", GOVERNANCE],
        ],
        [3000, 6972],
    )
    d.h1("Project Governance Portfolio -- folder contents")
    d.para("All folders below are complete in the Project Governance Portfolio (confirmed by the team, 13 October 2026).")
    d.table(
        ["Folder / item", "Contents"],
        [
            ["Project Proposal", "Inclusive_Auslan_Learning_Assistant_Proposal.pptx"],
            ["Industry Mentor Pitch", "In the Portfolio"],
            ["Persona", "Primary persona “Sarah Thompson” (see Product Document, Section 1)"],
            ["Ethics Canvas", "In the Portfolio"],
            ["System Architecture Diagrams", "website/docs/system-architecture.docx"],
            ["Iterations folder (1-3)", "Presentation, Iteration Analysis & Design Report, Data Management Plan, Agenda/Minutes/Reflection, Usability Video, Retrospectives, per iteration"],
            ["Code and Database folder", "schema.sql, server/data/signs.json, server/data/sign_videos.json, avatar motion clips (sign_poses), recognition/models, HandMirror-avatar-build-scripts.zip, Final Data Management Plan"],
            ["Project Support folder", "Product video, stand-up notes, Expo content, team info doc"],
            ["Security Folder", "Security Plan, Vulnerability Assessment Report"],
            ["Handover Documents", "01-Product-Document.docx, 02-Support-Document.docx, 03-Maintenance-Document.docx (this folder); previous versions in handover/archive/"],
            ["Code / backup", "GitHub repository: " + REPO],
        ],
        [3000, 6972],
    )
    d.h1("Sign-off")
    d.para("Team TE38 (“BitDance”)")
    d.table(["Name", "Signature", "Date"], [[m[0], "", ""] for m in TEAM], [3800, 3800, 2372])
    d.h2("Monash mentor acceptance")
    d.table(["Name", "Signature", "Date"], [["Afsana", "", ""], ["Ted", "", ""]], [3800, 3800, 2372])
    d.closing("HandMirror — Handover Checklist and Sign-Off · Team TE38 (BitDance) · FIT5120")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    for name, build in [
        ("01-Product-Document.docx", product),
        ("02-Support-Document.docx", support),
        ("03-Maintenance-Document.docx", maintenance),
        ("04-Handover-Checklist-and-Sign-off.docx", checklist),
    ]:
        doc = Doc(root / name)
        build(doc)
        doc.save()
        print("wrote", name)
