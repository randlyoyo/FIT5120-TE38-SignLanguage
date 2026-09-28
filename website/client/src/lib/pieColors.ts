// A curated palette (not the tag-hash hues used elsewhere -- those can put
// two near-identical colors next to each other by chance) ordered around
// the hue wheel, one full turn in ten roughly-even steps.
const PALETTE = [
  "#9c2f2f", // brick
  "#9c5b2f", // rust
  "#a3760f", // amber
  "#7a9c2f", // olive
  "#3a7a4a", // green
  "#2f7a7a", // teal
  "#3a5b7a", // steel blue
  "#1f3d7a", // blue (site accent)
  "#5b5ba3", // indigo
  "#8a3a6b", // plum
];

// Coprime with PALETTE.length, so walking the palette by this step visits
// every color before repeating, and consecutive picks always land ~3
// slots (~110 degrees of hue) apart -- neighbouring wedges in the pie
// never end up close in hue.
const STEP = 3;

/** One color per wedge index, in wedge order -- consecutive indices are
 *  the pie's actual neighbours, so this is what keeps them visually
 *  distinct instead of just assigning colors independently per tag. */
export function pieColors(count: number): string[] {
  return Array.from({ length: count }, (_, i) => PALETTE[(i * STEP) % PALETTE.length]);
}
