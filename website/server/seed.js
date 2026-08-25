require("dotenv").config();
const pool = require("./src/db");

// Clearly-labeled placeholder signs -- NOT real Auslan words. See
// README.md for the "Placeholder / Demo Data" source note.
// `definitions` uses a single neutral "Meaning" group rather than
// inventing grammatical senses for a made-up gesture; `keywords` stay
// generic (e.g. "relative", not "dad") so they never imply a specific
// real meaning that hasn't been verified.
const DEMO_SIGNS = [
  {
    gloss: "DEMO SIGN A",
    tags: ["greeting"],
    keywords: ["hi", "hey", "wave hello"],
    definitions: [
      { partOfSpeech: "Meaning", senses: ["Placeholder stand-in for a friendly hello-style greeting."] },
    ],
    usage_notes: [
      "Raise hand to shoulder height, palm facing forward.",
      "Move hand in a small side-to-side wave.",
      "Hold a relaxed, friendly expression.",
    ],
  },
  {
    gloss: "DEMO SIGN B",
    tags: ["greeting"],
    keywords: ["bye", "goodbye", "see you"],
    definitions: [
      { partOfSpeech: "Meaning", senses: ["Placeholder stand-in for a farewell-style greeting."] },
    ],
    usage_notes: [
      "Raise hand, palm facing outward.",
      "Close fingers toward palm once, like a gentle wave goodbye.",
    ],
  },
  {
    gloss: "DEMO SIGN C",
    tags: ["greeting"],
    keywords: ["thanks", "cheers", "much obliged"],
    definitions: [
      { partOfSpeech: "Meaning", senses: ["Placeholder stand-in for a polite thank-you gesture."] },
    ],
    usage_notes: [
      "Touch fingertips to chin.",
      "Move hand forward and slightly down toward the other person.",
    ],
  },
  {
    gloss: "DEMO SIGN D",
    tags: ["family"],
    keywords: ["relative", "family member"],
    definitions: [
      { partOfSpeech: "Meaning", senses: ["Placeholder stand-in for a family-member sign."] },
    ],
    usage_notes: ["Form a loose fist near the temple.", "Tap gently twice."],
  },
  {
    gloss: "DEMO SIGN E",
    tags: ["family"],
    keywords: ["relative", "family member"],
    definitions: [
      { partOfSpeech: "Meaning", senses: ["Placeholder stand-in for another family-member sign."] },
    ],
    usage_notes: ["Extend thumb from a loose fist near the chin.", "Hold briefly."],
  },
  {
    gloss: "DEMO SIGN F",
    tags: ["family"],
    keywords: ["relative", "family member"],
    definitions: [
      { partOfSpeech: "Meaning", senses: ["Placeholder stand-in for a third family-member sign."] },
    ],
    usage_notes: ["Extend pinky finger from a loose fist near the chin.", "Hold briefly."],
  },
  {
    gloss: "DEMO SIGN G",
    tags: ["action"],
    keywords: ["give", "offer"],
    definitions: [
      { partOfSpeech: "Meaning", senses: ["Placeholder stand-in for an everyday action verb."] },
    ],
    usage_notes: [
      "Both hands flat, palms up.",
      "Move hands slightly forward and up, as if offering something.",
    ],
  },
  {
    gloss: "DEMO SIGN H",
    tags: ["action"],
    keywords: ["point", "indicate"],
    definitions: [
      { partOfSpeech: "Meaning", senses: ["Placeholder stand-in for a second action verb."] },
    ],
    usage_notes: ["Point index finger forward.", "Move hand in a small forward arc."],
  },
  {
    gloss: "DEMO SIGN I",
    tags: ["action"],
    keywords: ["turn", "rotate"],
    definitions: [
      { partOfSpeech: "Meaning", senses: ["Placeholder stand-in for a third action verb."] },
    ],
    usage_notes: ["Both hands loosely closed.", "Rotate wrists alternately, like turning a small wheel."],
  },
];

async function seed() {
  for (const s of DEMO_SIGNS) {
    await pool.query(
      `INSERT INTO signs (gloss, definitions, usage_notes, tags, keywords, source)
       VALUES (?, CAST(? AS JSON), CAST(? AS JSON), CAST(? AS JSON), CAST(? AS JSON), 'Placeholder / Demo Data -- Not Real Auslan')
       ON DUPLICATE KEY UPDATE
         definitions = VALUES(definitions), usage_notes = VALUES(usage_notes),
         tags = VALUES(tags), keywords = VALUES(keywords)`,
      [
        s.gloss,
        JSON.stringify(s.definitions),
        JSON.stringify(s.usage_notes),
        JSON.stringify(s.tags),
        JSON.stringify(s.keywords),
      ]
    );
  }
  console.log(`Seeded ${DEMO_SIGNS.length} placeholder signs.`);
  await pool.end();
}

seed().catch((err) => {
  console.error(err);
  process.exit(1);
});
