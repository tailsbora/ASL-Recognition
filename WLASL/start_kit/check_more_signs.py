import json

TARGET_SIGNS = [
    "hello", "help", "love", "me", "no",
    "please", "sorry", "thank you", "yes", "you",

    "good", "bad", "want", "need", "like",
    "know", "think", "understand",

    "eat", "drink", "water", "food",

    "home", "school", "work",

    "friend", "family",
    "mother", "father",
    "brother", "sister",

    "man", "woman", "child",

    "happy", "sad", "angry", "tired",

    "morning", "night",
    "today", "tomorrow", "yesterday",

    "what", "where", "who",
    "why", "how", "when",

    "go", "come", "stop", "wait",

    "more", "again", "finished",

    "name", "time", "money",

    "bathroom",
    "car", "phone", "computer",

    "hot", "cold",
    "big", "small",

    "right", "wrong",
    "here", "there"
]

with open("WLASL_v0.3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

available = {}

for entry in data:
    gloss = entry["gloss"].lower().strip()

    if gloss in TARGET_SIGNS:
        available[gloss] = len(entry["instances"])

print()
print("AVAILABLE SIGNS")
print("=" * 45)

for sign in TARGET_SIGNS:
    if sign in available:
        print(f"{sign:15} {available[sign]:3} videos")

print()
print("NOT FOUND")
print("=" * 45)

for sign in TARGET_SIGNS:
    if sign not in available:
        print(sign)