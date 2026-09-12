import { createIdListStore } from "./idListStore";

const store = createIdListStore("auslan-website.toLearnSigns.v1");

export const getToLearnIds = store.getIds;
export const isToLearn = store.has;
export const toggleToLearn = store.toggle;
export const removeFromToLearn = store.remove;
export const addAllToLearn = store.addAll;
