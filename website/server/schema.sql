-- Auslan Inclusive Learning Assistant -- Sign Library schema
-- Usage: mysql -u root -p < schema.sql

CREATE DATABASE IF NOT EXISTS auslan_learning
  CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;

USE auslan_learning;

CREATE TABLE IF NOT EXISTS signs (
  id                 INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,

  -- Sign label/gloss shown as the card title, e.g. "DEMO SIGN A" for
  -- placeholders, a real English word once verified content exists.
  gloss              VARCHAR(100)  NOT NULL,

  -- Structured, Auslan-Signbank-style definition: grouped by part of
  -- speech, each group holding one or more distinct senses, e.g.
  --   [{"partOfSpeech":"Noun","senses":["...","..."]},
  --    {"partOfSpeech":"Verb or Adjective","senses":["..."]}]
  -- Placeholder rows use a single neutral "Meaning" group rather than
  -- inventing grammatical senses for a made-up gesture.
  definitions        JSON          NOT NULL,

  -- Ordered array of short strings describing how to perform/use the sign,
  -- e.g. ["Step 1: ...", "Step 2: ..."]. Native JSON so mysql2 returns it
  -- already parsed, and a DS teammate can SELECT/inspect it directly.
  usage_notes        JSON          NOT NULL,

  -- Multi-value classification tags -- replaces a single `category` column,
  -- which could only ever put a sign in one group at a time and couldn't
  -- represent signs that genuinely belong to more than one group, e.g.
  -- ["family","greeting"]. Rendered as visible chips in the UI and used
  -- for browsing/filtering.
  tags               JSON          NULL,

  -- Alternate search terms/synonyms a user might type that are NOT a
  -- classification, e.g. ["hi","hey"] for a sign glossed HELLO. Search-only:
  -- never rendered as a visible tag, so "what this is classified as" and
  -- "other words people might search for it" don't get conflated.
  keywords           JSON          NULL,

  -- Provenance note, e.g. "Placeholder / Demo Data -- Not Real Auslan" or
  -- "Auslan Signbank (pending verification)".
  source             VARCHAR(150)  NULL,

  created_at         TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at         TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP
                     ON UPDATE CURRENT_TIMESTAMP,

  UNIQUE KEY uq_signs_gloss (gloss)
) ENGINE=InnoDB;
