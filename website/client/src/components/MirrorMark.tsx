interface Props {
  size?: number;
}

/** Small mark echoing the favicon: a hand, and its mirrored reflection. */
export function MirrorMark({ size = 22 }: Props) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <line x1="3" y1="13.5" x2="21" y2="13.5" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.35" />
      <g fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <rect x="9.4" y="8.5" width="5.2" height="4.5" rx="1.4" />
        <line x1="10.2" y1="8.5" x2="10.2" y2="4.8" />
        <line x1="11.6" y1="8.5" x2="11.6" y2="3.6" />
        <line x1="13.0" y1="8.5" x2="13.0" y2="3.8" />
        <line x1="14.4" y1="8.5" x2="14.4" y2="5.2" />
        <line x1="9.4" y1="10.5" x2="7.6" y2="9.2" />
      </g>
      <g fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" opacity="0.4">
        <rect x="9.4" y="14" width="5.2" height="4.5" rx="1.4" />
        <line x1="10.2" y1="18.5" x2="10.2" y2="22.2" />
        <line x1="11.6" y1="18.5" x2="11.6" y2="23.4" />
        <line x1="13.0" y1="18.5" x2="13.0" y2="23.2" />
        <line x1="14.4" y1="18.5" x2="14.4" y2="21.8" />
        <line x1="9.4" y1="16.5" x2="7.6" y2="17.8" />
      </g>
    </svg>
  );
}
