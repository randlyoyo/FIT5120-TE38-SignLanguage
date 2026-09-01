import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useSigns } from "../context/SignsContext";
import { loadProgress, type ProgressMap } from "../lib/storage";

export function Progress() {
  const { signs, loading } = useSigns();
  const [progress, setProgress] = useState<ProgressMap>({});

  useEffect(() => {
    setProgress(loadProgress());
  }, []);

  if (loading) return <p>Loading…</p>;

  const rows = signs
    .map((sign) => {
      const entry = progress[sign.id];
      const accuracy = entry && entry.attempts > 0 ? entry.correct / entry.attempts : null;
      return { sign, entry, accuracy };
    })
    .sort((a, b) => (a.accuracy ?? -1) - (b.accuracy ?? -1));

  return (
    <section>
      <h1>Your progress</h1>
      <p>Based on quiz answers and webcam practice attempts stored only on this device.</p>

      <table className="progress-table">
        <thead>
          <tr>
            <th>Sign</th>
            <th>Attempts</th>
            <th>Accuracy</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ sign, entry, accuracy }) => (
            <tr key={sign.id}>
              <td>{sign.label}</td>
              <td>{entry?.attempts ?? 0}</td>
              <td>{accuracy === null ? "—" : `${Math.round(accuracy * 100)}%`}</td>
              <td>
                <Link to={`/learn/${sign.id}`}>Review</Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {rows.every((r) => !r.entry) && <p>No practice or quiz attempts recorded yet — try the quiz or practice pages.</p>}
    </section>
  );
}
