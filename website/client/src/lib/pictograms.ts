// Human Pictogram 2.0 (pictogram2.com, TopeconHeroes) -- free for use as part
// of a design, not for redistribution on their own. Recoloured from the
// original green to the site accent blue; files are named by pictogram code.
const files = import.meta.glob<string>("../assets/pictograms/*.png", { eager: true, import: "default" });

function pictogram(code: string): string {
  const url = files[`../assets/pictograms/${code}.png`];
  if (!url) throw new Error(`missing pictogram ${code}`);
  return url;
}

const SCENARIO_CODES: Record<string, string> = {
  A01: "p0462", A02: "p0182", A03: "p0382", A04: "p0636", A05: "p0358", A06: "p0974",
  A07: "p0938", A08: "p0737", A09: "p0332", A10: "p0661", A11: "p0919", A12: "p0004",
  A13: "p0996", A14: "p0300", A15: "p0354", A16: "p0429", A17: "p0941", A18: "p0353",
  A19: "p0250", A20: "p0559", A21: "p0461", A22: "p0459", A23: "p0016", A24: "p0844",
  A25: "p0357", A26: "p0920", A27: "p0583", B01: "p0267", B02: "p0117", B03: "p0793",
  B04: "p0887", B05: "p0618", C01: "p0479", C02: "p0713",
};

/** Pictogram for a scenario journey (A01..C02). */
export const scenarioPictogram = (id: string) => pictogram(SCENARIO_CODES[id] ?? "p0417");

/** Pictograms for the three scenario sections. */
export const SECTION_PICTOGRAMS = { A: pictogram("p0636"), B: pictogram("p0671"), C: pictogram("p0417") };

/** Marks a message the user signed on camera (Sign Chat). */
export const SIGNED_PICTOGRAM = pictogram("p0357");

/** The end of a journey's route (trophy). */
export const FINISH_PICTOGRAM = pictogram("p0793");
