import { createIdListStore } from "./idListStore";

const store = createIdListStore("auslan-website.learnedSigns.v1");

export const getLearnedIds = store.getIds;
export const isLearned = store.has;
export const toggleLearned = store.toggle;
