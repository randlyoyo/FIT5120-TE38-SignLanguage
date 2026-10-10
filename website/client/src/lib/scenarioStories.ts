import type { SignLevel } from "../api/types";

export interface ScenarioStep {
  title: string;
  /** Glosses this step teaches -- each one verified against the live
   *  library as an exact gloss match (see scripts/checkScenarioWords.mjs),
   *  so every chip lands on a real sign's detail page, not a search
   *  fallback. */
  words: string[];
  /** A short example sentence for this step, shown as context -- not sent
   *  anywhere, just read. */
  phrase: string;
}

export interface Scenario {
  id: string;
  emoji: string;
  title: string;
  level: SignLevel;
  /** The one-line setup shown as the "Scenario:" quote. */
  situation: string;
  steps: ScenarioStep[];
}

/**
 * Content plan from the team's own scenario-design notes -- a scenario is a
 * short story walked through step by step, each step teaching a handful of
 * signs, rather than a flat vocabulary list.
 */
export const SCENARIOS: Scenario[] = [
  {
    id: "meeting-someone-new",
    emoji: "👋",
    title: "Meeting Someone New",
    level: "beginner",
    situation: "You meet a new person at school.",
    steps: [
      { title: "Say hello", words: ["HELLO", "HI"], phrase: "Hello!" },
      { title: "Meet them", words: ["MEET", "FRIEND"], phrase: "I'd like you to meet my friend." },
      { title: "Say it's nice to meet them", words: ["NICE"], phrase: "Nice to meet you." },
      { title: "Ask a question", words: ["QUESTION"], phrase: "Can I ask you a question?" },
      { title: "Say thanks", words: ["THANKS"], phrase: "Thanks for chatting!" },
    ],
  },
  {
    id: "ordering-at-a-cafe",
    emoji: "☕",
    title: "Ordering at a Café",
    level: "beginner",
    situation: "You're ordering a drink at a café.",
    steps: [
      { title: "Arrive at the café", words: ["CAFE"], phrase: "Let's go to the café." },
      { title: "Order a tea", words: ["TEA"], phrase: "I would like a tea." },
      { title: "Ask the price", words: ["PRICE"], phrase: "How much is it?" },
      { title: "Pay", words: ["PAY"], phrase: "I will pay now." },
      { title: "Say thanks", words: ["THANKS"], phrase: "Thank you!" },
    ],
  },
  {
    id: "asking-for-directions",
    emoji: "🗺️",
    title: "Asking for Directions",
    level: "intermediate",
    situation: "You're lost and need to find your way.",
    steps: [
      { title: "Ask for help", words: ["HELP"], phrase: "Can you help me?" },
      { title: "Name the street", words: ["STREET"], phrase: "I'm looking for this street." },
      { title: "Follow directions", words: ["STRAIGHT"], phrase: "Go straight ahead." },
      { title: "Check the distance", words: ["NEAR", "FAR"], phrase: "Is it near or far?" },
      { title: "Find it", words: ["FIND"], phrase: "I found it!" },
    ],
  },
  {
    id: "at-the-doctor",
    emoji: "🩺",
    title: "At the Doctor",
    level: "intermediate",
    situation: "You're not feeling well and visit the doctor.",
    steps: [
      { title: "Say you're sick", words: ["SICK"], phrase: "I feel sick." },
      { title: "See the doctor", words: ["DOCTOR"], phrase: "I need to see a doctor." },
      { title: "Describe the pain", words: ["HEAD"], phrase: "My head hurts." },
      { title: "Meet the nurse", words: ["NURSE"], phrase: "The nurse will check you." },
      { title: "Feel better", words: ["BETTER"], phrase: "I feel better now." },
    ],
  },
  {
    id: "shopping-for-clothes",
    emoji: "👕",
    title: "Shopping for Clothes",
    level: "beginner",
    situation: "You're shopping for a new shirt.",
    steps: [
      { title: "Go to the shop", words: ["SHOP"], phrase: "Let's go to the shop." },
      { title: "Find a shirt", words: ["SHIRT"], phrase: "I'm looking for a shirt." },
      { title: "Check it fits", words: ["FIT"], phrase: "Does this fit me?" },
      { title: "Ask the price", words: ["PRICE"], phrase: "How much does it cost?" },
      { title: "Pay for it", words: ["PAY"], phrase: "I'll pay for it." },
    ],
  },
  {
    id: "catching-the-train",
    emoji: "🚆",
    title: "Catching the Train",
    level: "beginner",
    situation: "You're waiting for the train to go to the city.",
    steps: [
      { title: "Find the stop", words: ["STOP"], phrase: "Where is the train stop?" },
      { title: "Wait", words: ["WAIT"], phrase: "I will wait here." },
      { title: "Buy a ticket", words: ["TICKET"], phrase: "One ticket, please." },
      { title: "Board the train", words: ["TRAIN"], phrase: "The train is here." },
      { title: "Find a seat", words: ["SEAT"], phrase: "Is this seat free?" },
    ],
  },
  {
    id: "making-weekend-plans",
    emoji: "📅",
    title: "Making Weekend Plans",
    level: "intermediate",
    situation: "You're planning what to do this weekend with a friend.",
    steps: [
      { title: "Bring up the weekend", words: ["WEEKEND"], phrase: "What are you doing this weekend?" },
      { title: "Suggest something fun", words: ["FUN"], phrase: "Let's do something fun." },
      { title: "Suggest playing", words: ["PLAY"], phrase: "Do you want to play?" },
      { title: "Arrange to meet", words: ["MEET"], phrase: "Let's meet up." },
      { title: "With a friend", words: ["FRIEND"], phrase: "Bring a friend too." },
    ],
  },
  {
    id: "job-interview",
    emoji: "💼",
    title: "A Job Interview",
    level: "advanced",
    situation: "You're attending a job interview.",
    steps: [
      { title: "Apply for the job", words: ["APPLY"], phrase: "I applied for this job." },
      { title: "Attend the interview", words: ["INTERVIEW"], phrase: "Thank you for the interview." },
      { title: "Describe your work", words: ["WORK"], phrase: "I enjoy my work." },
      { title: "Describe a skill", words: ["SKILL"], phrase: "This is one of my skills." },
      { title: "Finish with confidence", words: ["CONFIDENCE"], phrase: "I have confidence in myself." },
    ],
  },
  {
    id: "family-gathering",
    emoji: "🍽️",
    title: "Family Gathering",
    level: "intermediate",
    situation: "Your family is together at home.",
    steps: [
      { title: "Arrive home", words: ["HOME"], phrase: "We're home!" },
      { title: "Greet your mother", words: ["MOTHER"], phrase: "Hello, Mum." },
      { title: "Greet your father", words: ["FATHER"], phrase: "Hello, Dad." },
      { title: "Have lunch together", words: ["LUNCH"], phrase: "Let's have lunch together." },
      { title: "Celebrate with cake", words: ["CELEBRATION", "CAKE"], phrase: "Let's celebrate with cake!" },
    ],
  },
];
