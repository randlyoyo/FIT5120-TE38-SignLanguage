import { addAllLearned, removeLearned } from "./learnedSigns";
import { addAllToLearn, removeFromToLearn } from "./toLearnSigns";

// "To Learn" and "Learnt" are mutually exclusive: putting a sign on one list
// takes it off the other (same rule as the detail page's two toggles).

export function addToLearn(ids: number[]) {
  addAllToLearn(ids);
  ids.forEach(removeLearned);
}

export function markLearnt(ids: number[]) {
  addAllLearned(ids);
  ids.forEach(removeFromToLearn);
}
