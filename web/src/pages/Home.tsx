import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useSigns } from "../context/SignsContext";
import { getSignInfo } from "../lib/signs";

export function Home() {
  const { signs, loading, error } = useSigns();
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return signs.filter((sign) => {
      const info = getSignInfo(sign.id);
      const haystack = `${sign.label} ${info?.category ?? ""} ${info?.description ?? ""}`.toLowerCase();
      return q === "" || haystack.includes(q);
    });
  }, [signs, query]);

  return (
    <section>
      <h1>Sign library</h1>
      <p>Search for a sign, then open it to watch a guided demonstration or practise with your webcam.</p>

      <input
        type="search"
        className="search-input"
        placeholder="Search signs…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        aria-label="Search signs"
      />

      {loading && <p>Loading sign library…</p>}
      {error && <p role="alert">Couldn't load the sign library: {error}</p>}

      <ul className="sign-grid">
        {filtered.map((sign) => {
          const info = getSignInfo(sign.id);
          return (
            <li key={sign.id} className="sign-card">
              <h2>{sign.label}</h2>
              {info && <p className="category">{info.category}</p>}
              {info && <p>{info.description}</p>}
              <div className="sign-card-actions">
                <Link to={`/learn/${sign.id}`}>Learn</Link>
                <Link to={`/practice/${sign.id}`}>Practise</Link>
              </div>
            </li>
          );
        })}
      </ul>

      {!loading && filtered.length === 0 && <p>No signs match "{query}".</p>}
    </section>
  );
}
