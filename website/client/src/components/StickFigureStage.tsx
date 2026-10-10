interface Props {
  /** Small status pill above the stage, e.g. "Waiting for input" / "Generating…". */
  status?: string;
  /** Caption printed below the stage. */
  caption?: string;
  /** True while a reply is being "signed" -- swaps the idle sway for a
   *  faster gesture loop so the placeholder reads as active, not frozen. */
  signing?: boolean;
  className?: string;
}

/**
 * Placeholder stand-in for the motion-capture-driven avatar the generation
 * model will eventually render. An honest mockup, not a real skeleton --
 * abstract stick-figure line art (same construction as handGlyphs.tsx),
 * never a claim about a specific real sign.
 */
export function StickFigureStage({ status, caption, signing, className }: Props) {
  return (
    <div className={`stick-figure-stage ${className ?? ""}`}>
      {status && <span className="stick-figure-status">{status}</span>}
      <svg
        className={`stick-figure ${signing ? "signing" : "idle"}`}
        viewBox="0 0 120 160"
        fill="none"
        stroke="currentColor"
        strokeWidth="3.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        role="img"
        aria-label="Placeholder motion-capture avatar -- no generated animation yet"
      >
        <circle cx="60" cy="30" r="16" />
        <line x1="60" y1="46" x2="60" y2="95" />
        <line x1="60" y1="105" x2="42" y2="150" />
        <line x1="60" y1="105" x2="78" y2="150" />
        <line x1="60" y1="55" x2="30" y2="85" className="stick-figure-arm stick-figure-arm-l" />
        <line x1="60" y1="55" x2="90" y2="85" className="stick-figure-arm stick-figure-arm-r" />
      </svg>
      {caption && <p className="stick-figure-caption">{caption}</p>}
    </div>
  );
}
