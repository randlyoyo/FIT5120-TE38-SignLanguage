import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchSigns } from "../api/signs";

const FEATURES = [
  {
    title: "Search & Browse",
    body: "Look up a sign by gloss, synonym, or definition, or browse the full index by category.",
  },
  {
    title: "Demonstration & Learning",
    body: "Step through a sign's usage notes alongside its reference video to see how it's formed.",
  },
  {
    title: "Mark as Learned",
    body: "Stamp a sign as learned to track your own progress through the catalogue, right in your browser.",
  },
];

export function HomePage() {
  const [totalEntries, setTotalEntries] = useState<number | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchSigns({ pageSize: 1, signal: controller.signal })
      .then((data) => setTotalEntries(data.pagination.totalResults))
      .catch((err) => {
        if (err.name !== "AbortError") console.error(err);
      });
    return () => controller.abort();
  }, []);

  return (
    <div className="page-container">
      <section className="home-hero">
        <p className="library-eyebrow">Auslan sign language, indexed</p>
        <h1 className="page-title home-hero-title">HandMirror</h1>
        <p className="home-hero-tagline">
          A searchable Auslan sign reference, catalogued like a field
          specimen collection — look a sign up, see how it's formed, and
          track what you've learned.
        </p>
        <div className="home-hero-actions">
          <Link to="/library" className="home-cta">
            Browse the library &rarr;
          </Link>
          {totalEntries !== null && (
            <p className="results-count home-hero-count">{totalEntries} entries indexed</p>
          )}
        </div>
      </section>

      <section className="home-features">
        {FEATURES.map((feature) => (
          <div key={feature.title} className="home-feature-card">
            <h2 className="home-feature-title">{feature.title}</h2>
            <p className="home-feature-body">{feature.body}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
