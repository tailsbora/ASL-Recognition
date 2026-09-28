import json

TARGET_SIGNS = [
    "hello",
    "yes",
    "no",
    "please",
    "thank you",
    "sorry",
    "help",
    "love",
    "you",
    "me"
]

with open("WLASL_v0.3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

found = {}

for entry in data:
    gloss = entry["gloss"].lower().strip()

    if gloss in TARGET_SIGNS:
        found[gloss] = len(entry["instances"])

for sign in TARGET_SIGNS:
    if sign in found:
        print(sign, "->", found[sign], "videos")
    else:
        print(sign, "-> NOT FOUND")