import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import { Link, useNavigate } from "react-router-dom";
import { fetchSignById, fetchSigns } from "../api/signs";
import heroPoster from "../assets/home-hero-poster.webp";
import { SmplxAvatar } from "../components/SmplxAvatar";
import { SCENARIO_CATEGORIES } from "../lib/scenarioCategories";
import { scenarioPictogram } from "../lib/pictograms";
/** Hero footage: people signing, from Pexels (free licence, no attribution
 *  required). Played one after another from a random start, so only one clip
 *  downloads at a time. */
const HERO_CLIPS = ["10373883", "10374296", "10374298", "10373841"].map((id) => `/hero/${id}.mp4`);

/** Quick searches under the search box -- everyday first words. */
const TRY_WORDS = ["hello", "thank you", "family", "dog"];
/** The sign strip shows these family-friendly signs only -- a random
 *  pick from the whole catalogue can land on a crude or adult gloss. IDs of
 *  HELLO, THANK YOU, DOG, CAT, MOTHER, FATHER, FRIEND, PLAY, LOVE, BABY,
 *  SCHOOL, HOME, BIRD, BROTHER, SISTER, APPLE (all have a video). */
const FEATURED_SIGNS: [number, string][] = [
  [1606, "HELLO"], [1579, "THANK YOU"], [1929, "DOG"], [2503, "CAT"], [1069, "MOTHER"], [978, "FATHER"],
  [2308, "FRIEND"], [1879, "PLAY"], [3212, "LOVE"], [1860, "BABY"], [1530, "SCHOOL"], [1604, "HOME"],
  [2513, "BIRD"], [2784, "BROTHER"], [669, "SISTER"], [1387, "APPLE"],
];

/** The three result tiers Practice actually reports (routes/recognize.js). */
const SCORE_TIERS = [
  { score: 92, label: "Excellent! ✓", note: "80+ · a clear match", tone: "good" },
  { score: 68, label: "Almost there", note: "60–79 · recognisable", tone: "close" },
  { score: 41, label: "Keep practicing", note: "under 60 · try again", tone: "retry" },
] as const;

/** A strip that scrolls sideways on its own; two copies make the loop seamless. */
function Marquee({
  count,
  secondsPerCard,
  reverse,
  children,
}: {
  count: number;
  secondsPerCard: number;
  reverse?: boolean;
  children: (copy: number) => ReactNode;
}) {
  return (
    <div className="home3-marquee">
      {/* Duration scales with the card count so every strip moves at the same pace. */}
      <div className={`home3-marquee-track ${reverse ? "is-reverse" : ""}`} style={{ animationDuration: `${count * secondsPerCard}s` }}>
        {children(0)}
        {children(1)}
      </div>
    </div>
  );
}

/** True once the element has come within 400px of the viewport (and stays true). */
function useNearViewport<T extends Element>() {
  const ref = useRef<T | null>(null);
  const [near, setNear] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || near) return;
    const io = new IntersectionObserver(([e]) => e.isIntersecting && setNear(true), { rootMargin: "400px" });
    io.observe(el);
    return () => io.disconnect();
  }, [near]);
  return [ref, near] as const;
}

/** The (6 MB) avatar, mounted only once its folder comes near the viewport. */
function LazyAvatar() {
  const [ref, near] = useNearViewport<HTMLDivElement>();
  return (
    <div ref={ref} className="home3-avatar">
      {near && <SmplxAvatar clip={null} playing smile className="home3-avatar-canvas" />}
    </div>
  );
}

/** Sideways strip of signing videos; fetches the clips once it's near the viewport. */
function SignVideoStrip() {
  const [ref, near] = useNearViewport<HTMLDivElement>();
  const [urls, setUrls] = useState<Record<number, string>>({});
  useEffect(() => {
    if (!near) return;
    const controller = new AbortController();
    Promise.all(
      FEATURED_SIGNS.map(([id]) =>
        fetchSignById(id, controller.signal)
          .then((sign) => [id, sign.videos?.find((v) => v.videoUrl)?.videoUrl ?? ""] as const)
          .catch(() => [id, ""] as const)
      )
    ).then((pairs) => setUrls(Object.fromEntries(pairs.filter(([, u]) => u))));
    return () => controller.abort();
  }, [near]);
  return (
    <div ref={ref}>
      <Marquee count={FEATURED_SIGNS.length} secondsPerCard={7}>
        {(copy) =>
          FEATURED_SIGNS.map(([sid, gloss]) => (
            <Link
              key={`${copy}-${sid}`}
              to={`/signs/${sid}`}
              className="home3-sign-card"
              aria-hidden={copy === 1}
              tabIndex={copy === 1 ? -1 : undefined}
            >
              <span className="home3-sign-card-video">
                {urls[sid] && <video src={urls[sid]} autoPlay muted loop playsInline preload="auto" />}
              </span>
              <span className="home3-sign-card-gloss">{gloss}</span>
            </Link>
          ))
        }
      </Marquee>
    </div>
  );
}

type Tone = "blue" | "sun" | "leaf" | "clay" | "sky";

/** One coloured "folder" per feature: a tab with the section name, an open
 *  button, and a white card inside. Stacks over the previous one on scroll. */
function Folder({
  tone,
  number,
  eyebrow,
  title,
  to,
  cta,
  children,
}: {
  tone: Tone;
  number: string;
  eyebrow: string;
  title: string;
  to: string;
  cta: string;
  children: ReactNode;
}) {
  return (
    <section className={`home3-folder home3-folder--${tone}`}>
      <div className="home3-folder-head">
        <div>
          <p className="home3-folder-eyebrow">
            {number} · {eyebrow}
          </p>
          <h2 className="home3-folder-title">{title}</h2>
        </div>
        <Link to={to} className="home3-folder-open">
          <span>{cta}</span>
          <span className="home3-flower-button" aria-hidden="true">
            <span className="home3-flower-arrow">→</span>
          </span>
        </Link>
      </div>
      <div className="home3-folder-card">{children}</div>
    </section>
  );
}

/** A feature list: bold lead + one plain line each. */
function Points({ items }: { items: [string, string][] }) {
  return (
    <ul className="home3-points">
      {items.map(([lead, text]) => (
        <li key={lead}>
          <strong>{lead}</strong>
          <span>{text}</span>
        </li>
      ))}
    </ul>
  );
}

export function HomePage() {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [totalEntries, setTotalEntries] = useState<number | null>(null);
  const [clip, setClip] = useState(() => Math.floor(Math.random() * HERO_CLIPS.length));

  useEffect(() => {
    const controller = new AbortController();
    fetchSigns({ pageSize: 1, signal: controller.signal })
      .then((data) => setTotalEntries(data.pagination.totalResults))
      .catch((err: Error) => {
        if (err.name !== "AbortError") console.error(err);
      });
    return () => controller.abort();
  }, []);

  function search(e: FormEvent) {
    e.preventDefault();
    const q = query.trim();
    navigate(q ? `/library?query=${encodeURIComponent(q)}` : "/library");
  }

  const signCount = totalEntries !== null ? totalEntries.toLocaleString() : "3,000+";

  return (
    <div className="home3">
      {/* Full-bleed moving poster: people signing, duotoned into the brand blue. */}
      <section className="home3-hero" aria-label="People talking in sign language">
        <video
          key={clip}
          className="home3-hero-video"
          src={HERO_CLIPS[clip]}
          autoPlay
          muted
          playsInline
          preload="auto"
          onEnded={() => setClip((c) => (c + 1) % HERO_CLIPS.length)}
        />
        <div className="home3-hero-tint" aria-hidden="true" />
        {/* Five-second test: what this is, and who it's for, right next to the title. */}
        <div className="home3-hero-copy">
          <h1 className="home3-hero-mark">
            Learn Auslan
            <br />
            together
          </h1>
          <p className="home3-hero-lede">
            A free Auslan dictionary and practice tool for <strong>parents and families of Deaf and hard-of-hearing
            children</strong> — look up a sign, watch it, then practise it on camera.
          </p>
        </div>
        <a
          href="#start"
          className="home3-hero-scroll"
          aria-label="Scroll to start"
          onClick={(e) => {
            e.preventDefault();
            document.getElementById("start")?.scrollIntoView({ behavior: "smooth" });
          }}
        >
          <span className="home3-hero-scroll-arrow" aria-hidden="true">
            ↓
          </span>
        </a>
      </section>

      <section className="home3-statement" id="start">
        <div className="home3-statement-text">
          <p className="home3-kicker">A free Auslan dictionary for families</p>
          <h2 className="home3-title">Start with the word you need today.</h2>
          <p className="home3-poem">
            Type any English word and see it signed by a real person.
            <br />
            Slow it down, turn the 3D avatar, copy it together —
            <br />
            then try it on camera and see how close you got.
          </p>
          <form className="home3-search" onSubmit={search} role="search">
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Look up a word…"
              aria-label="Look up a word in the Auslan library"
            />
            <button type="submit">Look it up</button>
          </form>
          <p className="home3-try">
            Try:
            {TRY_WORDS.map((w) => (
              <Link key={w} to={`/library?query=${encodeURIComponent(w)}`}>
                {w}
              </Link>
            ))}
          </p>
        </div>
        <div className="home3-poster">
          <img
            src={heroPoster}
            alt="Poster: 'Language lives between us.' Two hands reaching toward each other, halftone-screened in blue and terracotta."
          />
        </div>
      </section>

      <div className="home3-folders">
        <Folder tone="blue" number="01" eyebrow="Sign Library" title="Look up any sign" to="/library" cta="Open the library">
          <div className="home3-split home3-library-split">
            <div>
              <p className="home3-big-number">
                {signCount}
                <span>Auslan signs, each with a real signer on video</span>
              </p>
              <p className="home3-card-text">
                Search in English or browse by topic. Every sign has a video you can slow down, a step-by-step breakdown
                and a 3D avatar you can turn.
              </p>
            </div>
            <SignVideoStrip />
          </div>
        </Folder>

        <Folder tone="clay" number="02" eyebrow="Practice" title="Sign it back to the camera" to="/library" cta="Find a sign to practise">
          <div className="home3-split">
            <ol className="home3-steps">
              <li>
                <strong>Open a sign</strong>
                <span>Pick any sign with a Practice tab.</span>
              </li>
              <li>
                <strong>Sign it at your webcam</strong>
                <span>Video stays on your device — only hand positions are checked.</span>
              </li>
              <li>
                <strong>Get your score</strong>
                <span>One of three results, each with a tip on what to try next.</span>
              </li>
            </ol>
            <div className="home3-scores" aria-label="The three practice results">
              {SCORE_TIERS.map((t) => (
                <div key={t.label} className={`home3-score is-${t.tone}`}>
                  <span className="home3-score-ring" style={{ "--pct": `${t.score}%` } as React.CSSProperties}>
                    <strong>{t.score}</strong>
                  </span>
                  <span className="home3-score-label">{t.label}</span>
                  <span className="home3-score-note">{t.note}</span>
                </div>
              ))}
            </div>
          </div>
        </Folder>

        <Folder tone="sky" number="03" eyebrow="Sign Chat" title="Chat with a signing avatar" to="/conversation" cta="Try Sign Chat">
          <div className="home3-split home3-chat-split">
            <div className="home3-chat-intro">
              <p className="home3-chat-headline">
                Ask in English.
                <br />
                Get the answer in Auslan.
              </p>
              <p className="home3-card-text">
                Type a sentence or sign one at your camera — the avatar signs back, word by word.
              </p>
            </div>
            <div className="home3-chat-stage">
              <div className="home3-chat" aria-hidden="true">
                <span className="home3-bubble home3-bubble--me">Where is the bus stop?</span>
                <span className="home3-bubble home3-bubble--avatar">The bus stop is over there.</span>
              </div>
              <LazyAvatar />
            </div>
          </div>
        </Folder>

        <Folder tone="leaf" number="04" eyebrow="Scenario Learning" title="Signs for real situations" to="/scenarios" cta="See all journeys">
          <p className="home3-card-text home3-card-lead">
            {SCENARIO_CATEGORIES.length} journeys, from ordering at a café to seeing a doctor. Each one is split into short
            stops, so you always know what to learn next.
          </p>
          <Marquee count={SCENARIO_CATEGORIES.length} secondsPerCard={5} reverse>
            {(copy) =>
              SCENARIO_CATEGORIES.map((g) => (
                <Link
                  key={`${copy}-${g.id}`}
                  to={`/scenarios/category/${g.id}`}
                  className="home3-journey"
                  aria-hidden={copy === 1}
                  tabIndex={copy === 1 ? -1 : undefined}
                >
                  <img src={scenarioPictogram(g.id)} alt="" />
                  <span>{g.title}</span>
                </Link>
              ))
            }
          </Marquee>
        </Folder>

        <Folder tone="sun" number="05" eyebrow="Your lists" title="Keep track as you learn" to="/to-learn" cta="Open To Learn">
          <Points
            items={[
              ["To Learn", "Queue the signs you want to work on, grouped by when you added them."],
              ["Personalised sessions", "Pick topics and a word count — we fill your list with signs you haven't learnt."],
              ["Learnt", "Mark a sign learnt and watch your progress grow across every journey."],
            ]}
          />
        </Folder>
      </div>
    </div>
  );
}
