import { Link, NavLink } from "react-router-dom";
import { MirrorMark } from "./MirrorMark";

export function SiteHeader() {
  return (
    <header className="site-header">
      <div className="site-header-inner">
        <Link to="/" className="site-brand">
          <MirrorMark size={30} />
          <span>HandMirror</span>
        </Link>
        <nav className="site-nav">
          <NavLink to="/library" className={({ isActive }) => (isActive ? "active" : "")}>
            Library
          </NavLink>
          <NavLink to="/to-learn" className={({ isActive }) => (isActive ? "active" : "")}>
            To Learn
          </NavLink>
          <NavLink to="/learned" className={({ isActive }) => (isActive ? "active" : "")}>
            Learned
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
