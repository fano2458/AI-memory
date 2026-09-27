import random

from .world import CANCEL, REFINE, REVERT, SET, SUPERSEDE, TEMPORARY, UNCERTAIN

GENERIC = {
    SET: [
        "{poss} {rel} is {val}.",
        "Just so you know, {rel} is {val} for me.",
        "Note: {poss} {rel} is {val}.",
    ],
    SUPERSEDE: [
        "{poss} {rel} is {val} from here on.",
        "Going forward {poss} {rel} is {val}.",
        "Make it {val} for {poss} {rel}.",
    ],
    TEMPORARY: [
        "For a short stretch {poss} {rel} is {val}.",
        "Briefly, {rel} is {val}, then back to normal.",
    ],
    REVERT: [
        "Back to {val} for {poss} {rel}.",
        "{poss} {rel} is {val} again.",
    ],
    REFINE: [
        "To be precise, {poss} {rel} is {val}.",
        "More specifically, {val}.",
    ],
    UNCERTAIN: [
        "{subj} maybe making {poss} {rel} {val}, still deciding.",
        "There is a chance {rel} becomes {val}, nothing settled.",
        "{subj} weighing up {val} for {poss} {rel}.",
    ],
    CANCEL: [
        "{poss} {rel} is not settled any more, disregard what I said.",
        "Scrap the {rel} given earlier.",
    ],
}

BY_RELATION = {
    "location": {
        SET: ["{subj} based in {val}.", "{subj} living in {val} these days."],
        SUPERSEDE: ["{subj} moved to {val}.", "{subj} settled into {val}."],
        TEMPORARY: ["{subj} in {val} for a few days.", "{subj} staying in {val} briefly."],
        REVERT: ["{subj} back in {val}.", "{subj} moved back to {val}."],
        UNCERTAIN: ["{subj} maybe moving to {val}.", "{subj} possibly relocating to {val}."],
    },
    "job_title": {
        SET: ["{subj} working as a {val}.", "{poss} role is {val}."],
        SUPERSEDE: ["{subj} taken a {val} role.", "{subj} moved into a {val} position."],
        UNCERTAIN: ["There's talk of me becoming a {val}.", "I may move to a {val} role."],
    },
    "deadline": {
        SET: ["The deadline is {val}.", "We're due {val}."],
        SUPERSEDE: ["The deadline moved to {val}.", "They pushed it to {val}."],
        TEMPORARY: ["It's provisionally {val} for now."],
        REVERT: ["The deadline is {val} again."],
        UNCERTAIN: ["They may shift the deadline to {val}."],
    },
    "api_version": {
        SET: ["We're on {val} of the API.", "The service runs {val}."],
        SUPERSEDE: ["We migrated the API to {val}.", "We cut over to {val}."],
        REVERT: ["We rolled the API back to {val}."],
        UNCERTAIN: ["We might upgrade the API to {val}."],
    },
    "destination": {
        SET: ["{subj} heading to {val}.", "The trip is to {val}."],
        SUPERSEDE: ["{subj} changed the trip to {val}.", "{subj} rebooked for {val}."],
        UNCERTAIN: ["{subj} maybe going to {val} instead."],
    },
}

NOISE = [
    "Any tips for keeping houseplants alive in winter?",
    "I've been trying to cut down on coffee lately.",
    "Do you know a good way to organise reference notes?",
    "The weather has been unusually mild this week.",
    "I finally finished that book I kept putting off.",
    "Thinking of trying a new recipe this weekend.",
]


def render(event, rng=None):
    rng = rng or random.Random(event.event_id)
    by_rel = BY_RELATION.get(event.relation, {})
    options = by_rel.get(event.op) or GENERIC.get(event.op) or GENERIC[SET]
    rel = event.relation.replace("_", " ")
    first = event.entity == "user"
    poss = "My" if first else f"{event.entity}'s"
    subj = "I'm" if first else f"{event.entity} is"
    text = rng.choice(options).format(rel=rel, val=event.value, poss=poss, subj=subj)
    if not first and poss not in text and event.entity not in text:
        text = f"{event.entity}: {text}"
    return text


def noise_turn(rng):
    return rng.choice(NOISE)
