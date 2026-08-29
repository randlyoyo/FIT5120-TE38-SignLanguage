import { Link } from "react-router-dom";
import type { TagCount } from "../api/types";

interface Props {
  tags: TagCount[];
  activeTag: string;
}

/** Left-rail index of every category, like a library shelf directory --
    the primary way to browse the catalogue by topic. */
export function CategoryRail({ tags, activeTag }: Props) {
  if (tags.length === 0) return null;

  return (
    <nav className="category-rail" aria-label="Browse by category">
      <p className="category-rail-heading">Index</p>
      <ul>
        <li>
          <Link to="/" className={!activeTag ? "active" : ""}>
            All entries
          </Link>
        </li>
        {tags.map(({ tag, count }) => (
          <li key={tag}>
            <Link
              to={`/?tag=${encodeURIComponent(tag)}`}
              className={activeTag === tag ? "active" : ""}
            >
              <span>{tag}</span>
              <span className="category-rail-count">{count}</span>
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}
