import random

from .world import CANCEL, REFINE, REVERT, SET, SUPERSEDE, TEMPORARY, UNCERTAIN

GENERIC = {
    SET: [
        "My {rel} is {val}.",
        "Just so you know, {rel} is {val} for me.",
        "I should mention my {rel} is {val}.",
    ],
    SUPERSEDE: [
        "My {rel} is {val} from here on.",
        "Going forward my {rel} is {val}.",
        "Make it {val} for my {rel}.",
    ],
    TEMPORARY: [
        "For a short stretch my {rel} is {val}.",
        "Briefly, {rel} is {val}, then back to normal.",
    ],
    REVERT: [
        "Back to {val} for my {rel}.",
        "My {rel} is {val} again.",
    ],
    REFINE: [
        "To be precise, my {rel} is {val}.",
        "More specifically, {val}.",
    ],
    UNCERTAIN: [
        "I might make my {rel} {val}, still deciding.",
        "There is a chance {rel} becomes {val}, nothing settled.",
        "I am weighing up {val} for my {rel}.",
    ],
    CANCEL: [
        "My {rel} is not settled any more, disregard what I said.",
        "Scrap the {rel} I gave you.",
    ],
}

BY_RELATION = {
    "location": {
        SET: ["I'm based in {val}.", "I live in {val} these days."],
        SUPERSEDE: ["I've moved to {val}.", "I just settled into {val}."],
        TEMPORARY: ["I'm in {val} for a few days.", "Staying in {val} briefly."],
        REVERT: ["I'm back in {val}.", "Moved back to {val}."],
        UNCERTAIN: ["I might move to {val}.", "Thinking about relocating to {val}."],
    },
    "job_title": {
        SET: ["I work as a {val}.", "My role is {val}."],
        SUPERSEDE: ["I've taken a {val} role.", "I moved into a {val} position."],
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
        SET: ["I'm heading to {val}.", "The trip is to {val}."],
        SUPERSEDE: ["I changed the trip to {val}.", "I rebooked for {val}."],
        UNCERTAIN: ["I might go to {val} instead."],
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
    return rng.choice(options).format(rel=rel, val=event.value)


def noise_turn(rng):
    return rng.choice(NOISE)
