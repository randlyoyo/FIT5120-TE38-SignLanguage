import { Link } from "react-router-dom";
import type { TagCount } from "../api/types";

interface Props {
  tags: TagCount[];
  activeTag: string;
}

/** Left-rail index of every category, like a library shelf directory --
    the primary way to browse the catalogue by topic. Always renders the
    nav (never null) even before the tag list has loaded: this is the
    library layout's fixed-width grid column, and returning null while
    tags are still fetching used to collapse that column away entirely,
    squeezing the result grid into the rail's 240px track until the tags
    request resolved -- a real "single column, then three" layout jump on
    every fresh load, not just a loading-animation cosmetic issue. */
export function CategoryRail({ tags, activeTag }: Props) {
  return (
    <nav className="category-rail" aria-label="Browse by category">
      <p className="category-rail-heading">Index</p>
      <ul>
        <li>
          <Link to="/library" className={!activeTag ? "active" : ""}>
            All entries
          </Link>
        </li>
        {tags.length === 0
          ? Array.from({ length: 8 }).map((_, i) => (
              <li key={i} aria-hidden="true">
                <span className="skeleton-block category-rail-skeleton" />
              </li>
            ))
          : tags.map(({ tag, count }) => (
              <li key={tag}>
                <Link
                  to={`/library?tag=${encodeURIComponent(tag)}`}
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
