import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchSigns } from "../api/signs";
import heroPoster from "../assets/home-hero-poster.webp";
import { MirrorMark } from "../components/MirrorMark";
import { PlaceholderMedia } from "../components/PlaceholderMedia";
import { tagChipStyle } from "../lib/tagColors";
import type { Sign } from "../api/types";

export function HomePage() {
  const [totalEntries, setTotalEntries] = useState<number | null>(null);
  const [featured, setFeatured] = useState<Sign | null>(null);
  const [librarySample, setLibrarySample] = useState<Sign[]>([]);

  useEffect(() => {
    const controller = new AbortController();
    fetchSigns({ pageSize: 1, signal: controller.signal })
      .then((data) => {
        setTotalEntries(data.pagination.totalResults);
        const randomPage = Math.floor(Math.random() * data.pagination.totalResults) + 1;
        return fetchSigns({ page: randomPage, pageSize: 1, signal: controller.signal });
      })
      .then((data) => setFeatured(data.results[0] ?? null))
      .catch((err) => {
        if (err.name !== "AbortError") console.error(err);
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetchSigns({ pageSize: 4, signal: controller.signal })
      .then((data) => setLibrarySample(data.results))
      .catch((err) => {
        if (err.name !== "AbortError") console.error(err);
      });
    return () => controller.abort();
  }, []);

  return (
    <>
      <section className="home-hero-band">
        <div className="home-hero-inner">
          <div className="home-hero-text">
            <div className="home-lockup">
              <MirrorMark size={52} />
              <span>HandMirror</span>
            </div>
            <h1 className="home-hero-title">Look up a sign. Practise it. Teach your family.</h1>
            <p className="home-hero-tagline">
              HandMirror is a free Auslan sign dictionary built for families — search
              any word, watch how it's signed, and practise together with your kids
              at home.
            </p>
            <p className="home-hero-detail">
              It's just as useful for teachers and Auslan beginners: every entry pairs
              a real demonstration video with a clear, step-by-step breakdown of how
              the sign is formed.
            </p>
            <div className="home-hero-actions">
              <Link to="/library" className="home-cta">
                Browse the library
              </Link>
              {totalEntries !== null && (
                <p className="results-count home-hero-count">{totalEntries} signs indexed</p>
              )}
            </div>
          </div>

          <div className="home-video-block home-poster-block">
            <img
              className="home-poster-img"
              src={heroPoster}
              alt="Poster: 'Language lives between us.' Two hands reaching toward each other, halftone-screened in blue and terracotta. Auslan / Together. A free Auslan dictionary for every family."
            />
          </div>
        </div>
      </section>

      <div className="page-container">
        <section className="home-showcase-section">
          <div className="home-showcase-grid">
            <div className="home-showcase-text">
              <p className="home-showcase-kicker">01 — The library</p>
              <h2 className="home-section-heading">Search or browse, your way</h2>
              <p className="home-section-lead">
                Type any word, or explore the index by topic — Animals, Food &amp;
                Drink, Family, and more — so kids can browse just as easily as they
                can search.
              </p>
              <Link to="/library" className="home-showcase-link">
                Browse the library &rarr;
              </Link>
            </div>
            <div className="home-showcase-visual">
              <div className="home-library-preview">
                {librarySample.map((sign) => (
                  <Link key={sign.id} to={`/signs/${sign.id}`} className="home-library-preview-chip">
                    <span className="home-library-preview-gloss">{sign.gloss}</span>
                    {sign.tags[0] && (
                      <span className="tag-chip" style={tagChipStyle(sign.tags[0])}>
                        #{sign.tags[0]}
                      </span>
                    )}
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="home-showcase-section">
          <div className="home-showcase-grid home-showcase-grid-reverse">
            <div className="home-showcase-visual">
              {featured && (
                <div className="home-video-block">
                  <div className="home-video-frame">
                    <span className="home-video-accent home-video-accent-a" aria-hidden="true" />
                    <span className="home-video-accent home-video-accent-b" aria-hidden="true" />
                    <div className="home-video-media">
                      {featured.previewVideo?.videoUrl ? (
                        <video
                          src={featured.previewVideo.videoUrl}
                          autoPlay
                          muted
                          loop
                          playsInline
                          preload="auto"
                          aria-label={`${featured.gloss} Auslan demonstration`}
                        />
                      ) : (
                        <PlaceholderMedia seed={featured.id} gloss={featured.gloss} />
                      )}
                    </div>
                  </div>
                  <p className="home-video-caption home-video-caption-dark">
                    Practise today's sign <strong>{featured.gloss}</strong>
                  </p>
                </div>
              )}
            </div>
            <div className="home-showcase-text">
              <p className="home-showcase-kicker">02 — Every entry</p>
              <h2 className="home-section-heading">Watch it, practise it, mark it learned</h2>
              <p className="home-section-lead">
                Every sign pairs a real demonstration video with a clear,
                step-by-step breakdown of how it's formed — then stamp it
                "Learned" once your family has got it, right from that entry's
                page.
              </p>
              {featured && (
                <Link to={`/signs/${featured.id}`} className="home-showcase-link">
                  See the full entry for {featured.gloss} &rarr;
                </Link>
              )}
            </div>
          </div>
        </section>

        <section className="home-showcase-section">
          <div className="home-showcase-grid">
            <div className="home-showcase-text">
              <p className="home-showcase-kicker">03 — Coming soon</p>
              <h2 className="home-section-heading">Practise in front of the camera</h2>
              <p className="home-section-lead">
                We're building AI Auslan-sign recognition, so you'll be able to
                sign back at your camera and see straight away whether you've
                got it right.
              </p>
              <span className="home-coming-soon-badge">Coming soon</span>
            </div>
            <div className="home-showcase-visual">
              <AiRecognitionPreview />
            </div>
          </div>
        </section>
      </div>
    </>
  );
}

function AiRecognitionPreview() {
  return (
    <div className="home-ai-preview" aria-hidden="true">
      <svg viewBox="0 0 120 120" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
        <rect x="10" y="30" width="80" height="60" rx="8" />
        <path d="M90 48 L108 36 V84 L90 72" />
        <circle cx="50" cy="60" r="18" />
        <circle cx="50" cy="60" r="6" fill="currentColor" stroke="none" />
        <rect x="34" y="14" width="14" height="6" rx="2" />
      </svg>
    </div>
  );
}
