import { Link, NavLink } from "react-router-dom";
import { MirrorMark } from "./MirrorMark";

export function SiteHeader() {
  return (
    <header className="site-header">
      <div className="site-header-inner">
        <Link to="/" className="site-brand">
          <MirrorMark />
          <span>HandMirror</span>
        </Link>
        <nav className="site-nav">
          <NavLink to="/library" className={({ isActive }) => (isActive ? "active" : "")}>
            Library
          </NavLink>
          <NavLink to="/learned" className={({ isActive }) => (isActive ? "active" : "")}>
            Learned
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
