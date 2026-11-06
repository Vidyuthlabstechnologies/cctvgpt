"""SR-6: prompt filtering.

Blocks prompts that reference real people, real addresses/locations or real
incidents, and prohibited content categories (PRD section 5). The matcher is
deliberately conservative: if a prompt looks like it names someone or
somewhere real, it is refused and the user is asked to describe the scene
generically.

This is a rule-based first line of defence. The gazetteer and name lists in
safeguards/data/ are samples that the compliance owner must extend and
maintain [VERIFY: coverage review by the compliance owner], and a learned
classifier should be added before Phase 3. Note that the renderer only
understands a closed vocabulary (see prompt.py), so even a prompt that slips
through cannot make it depict a specific real person or place.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

CATEGORY_MESSAGES = {
    "evidence_misuse": "Outputs may not be used or presented as evidence, news or real footage.",
    "label_removal": "Requests to remove or hide synthetic markings are not permitted.",
    "real_person": "Prompts may not reference real or named people.",
    "real_location": "Prompts may not reference real addresses or named real places; describe the scene generically.",
    "real_incident": "Prompts may not recreate or reference real incidents.",
    "political_communal": "Political, religious or communal content is not permitted.",
    "sexual_content": "Sexual content is not permitted.",
    "minors": "Content depicting minors is not permitted.",
    "harassment": "Content targeting or impersonating individuals is not permitted.",
    "unsupported_violence": "Weapons and interpersonal violence are not in the v1 event library.",
}

_PATTERNS = {
    "evidence_misuse": [
        r"\bevidence\b", r"\bcourt\b", r"\bfir\b", r"\bpolice complaint\b", r"\bprove (?:that|he|she|they)\b",
        r"\bframe (?:him|her|them|someone|somebody)\b", r"\bincriminat\w*", r"\balibi\b", r"\bpass (?:it )?off as real\b",
        r"\blook(?:s)? (?:like )?(?:real|authentic|genuine)\b", r"\breal footage\b", r"\bauthentic footage\b",
        r"\bnews\b", r"\bviral\b", r"\bleak(?:ed)?\b", r"\bbreaking\b", r"\binsurance claim\b",
    ],
    "label_removal": [
        r"\b(?:no|without|remove|hide|delete|strip|crop out)\b.{0,20}\b(?:watermark|label|synthetic|marking|logo|badge|timestamp tag)\b",
        r"\bunlabel+ed\b",
    ],
    "sexual_content": [r"\bsex\w*", r"\bnud\w+", r"\bnaked\b", r"\bporn\w*", r"\berotic\w*", r"\bundress\w*", r"\bintimate\b", r"\bsensual\b"],
    "minors": [
        r"\bchild\w*", r"\bkids?\b", r"\bminors?\b", r"\bbab(?:y|ies)\b", r"\btoddlers?\b", r"\bteen\w*", r"\bschool ?(?:boy|girl|kid|child|student)s?\b",
        r"\bunder ?age\b", r"\b(?:boy|girl)s? (?:aged|of) \d", r"\b(?:[1-9]|1[0-7])[- ]?(?:year|yr)s?[- ]?old\b", r"\bjuvenile\w*",
    ],
    "political_communal": [
        r"\belection\w*", r"\bpoliti\w+", r"\brall(?:y|ies)\b", r"\bprotest\w*", r"\bcommunal\b", r"\bcaste\w*", r"\briots?\b",
        r"\b(?:hindu|muslim|christian|sikh|jain|buddhist|dalit)s?\b", r"\b(?:temple|mosque|masjid|church|gurudwara|gurdwara)s?\b",
        r"\b(?:bjp|congress|aap|tmc|dmk|aiadmk|shiv sena|cpi|cpm|rss|bsp|sp)\b", r"\bminister\w*", r"\bmla\b", r"\bmp\b",
    ],
    "harassment": [
        r"\bmy (?:ex|neighbou?r|boss|colleague|cowork\w+|landlord|tenant|teacher|wife|husband|girlfriend|boyfriend|classmate)\b",
        r"\bex[- ](?:wife|husband|girlfriend|boyfriend|partner)\b", r"\bharass\w*", r"\bblackmail\w*", r"\bdefam\w*",
        r"\brevenge\b", r"\bhumiliat\w*", r"\bimpersonat\w*", r"\bstalk\w*",
    ],
    "real_incident": [
        r"\b(?:26/11|9/11|7/7)\b", r"\b(?:recreate|reconstruct|re-?enact|replicate|reproduce)\b.{0,40}\b(?:incident|attack|accident|event|stampede|fire|crash|case)\b",
        r"\bwhat (?:really )?happened (?:at|in|on)\b", r"\b(?:terror\w*|bomb\w*|blast|massacre|hijack\w*)\b",
        r"\b(?:19|20)\d\d\b.{0,40}\b(?:attack|stampede|fire|accident|crash|incident|tragedy|disaster|collapse)\b",
        r"\b(?:attack|stampede|fire|accident|crash|incident|tragedy|disaster|collapse)\b.{0,40}\b(?:of|in|on) (?:19|20)\d\d\b",
    ],
    "real_location": [
        r"\b\d{1,5}[,\s]+(?:[a-z]+\s){0,3}(?:road|rd|street|st|lane|avenue|ave|marg|nagar|layout|cross|main|sector|block|colony|chowk)\b",
        r"\b\d{6}\b",  # Indian PIN code
        r"\bpin ?code\b", r"\bzip ?code\b", r"\baddress\b", r"\bgps\b", r"\b\d{1,2}\.\d{3,}\s*[ns]?,\s*\d{1,3}\.\d{3,}",
    ],
    "unsupported_violence": [
        r"\bguns?\b", r"\bpistols?\b", r"\brifles?\b", r"\bknif(?:e|es)\b", r"\bstab\w*", r"\bshoot\w*", r"\bshot (?:dead|down)\b",
        r"\bkill\w*", r"\bmurder\w*", r"\bblood\w*", r"\bgore\b", r"\bbeat(?:s|ing)? (?:up|him|her|someone)\b",
        r"\bfight\w*", r"\bassault\w*", r"\bkidnap\w*", r"\brobbery\b", r"\bweapons?\b", r"\bexplosi\w+",
    ],
}

# Capitalised words that are fine (scene vocabulary, camera labels, sentence starters).
_CAPS_ALLOW = {
    "cctv", "cam", "camera", "ir", "a", "an", "the", "at", "in", "on", "night", "day", "person", "man", "woman", "people",
    "someone", "parking", "lot", "warehouse", "corridor", "hallway", "office", "hospital", "lobby", "fire", "fall", "rain",
    "fog", "snow", "intrusion", "crowd", "abandoned", "bag", "dusk", "evening", "morning", "metro", "station", "airport",
    "factory", "mall", "store", "shop", "building", "gate", "entrance", "exit", "north", "south", "east", "west", "block",
    "level", "floor", "zone", "area", "i", "synthetic", "test", "scene", "low", "high", "wide", "angle", "view",
}


@dataclass
class FilterResult:
    allowed: bool
    categories: list = field(default_factory=list)
    matches: dict = field(default_factory=dict)

    @property
    def messages(self) -> list:
        return [CATEGORY_MESSAGES[c] for c in self.categories]

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "categories": self.categories, "matches": self.matches, "messages": self.messages}


@functools.lru_cache(maxsize=1)
def _lists():
    def load(name):
        p = DATA_DIR / name
        if not p.exists():
            return []
        return [ln.strip().lower() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]

    return load("place_names.txt"), load("person_names.txt")


def _normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("​", "")
    lower = t.lower()
    # Undo trivial obfuscation: s.e.x, n u d e, leetspeak.
    lower = re.sub(r"(?<=\b\w)[.\-_*](?=\w\b)", "", lower)
    lower = lower.translate(str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "@": "a", "$": "s"})) if re.search(r"[a-z][0-9@$][a-z]", lower) else lower
    return lower


def check_text(text: str) -> FilterResult:
    raw = text or ""
    t = _normalise(raw)
    res = FilterResult(True)

    def hit(cat, m):
        if cat not in res.categories:
            res.categories.append(cat)
        res.matches.setdefault(cat, []).append(m)

    for cat, pats in _PATTERNS.items():
        for p in pats:
            m = re.search(p, t)
            if m:
                hit(cat, m.group(0))
                break

    places, names = _lists()
    for place in places:
        if re.search(r"\b" + re.escape(place) + r"\b", t):
            hit("real_location", place)
            break
    for name in names:
        if re.search(r"\b" + re.escape(name) + r"\b", t):
            hit("real_person", name)
            break
    # Heuristic for proper names: two or more consecutive capitalised words that are
    # not scene vocabulary (e.g. "Ravi Kumar", "Brigade Road", "Phoenix Mall").
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", unicodedata.normalize("NFKC", raw)):
        words = [w.lower() for w in m.group(1).split()]
        if any(w not in _CAPS_ALLOW for w in words):
            hit("real_person" if not any(w in {"road", "street", "mall", "nagar", "station", "market", "park"} for w in words) else "real_location", m.group(1))
            break
    # Social handles, emails and phone numbers identify real people.
    for p in (r"(?<!\w)@\w{2,}", r"[\w.+-]+@[\w-]+\.\w+", r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b"):
        m = re.search(p, raw)
        if m:
            hit("real_person", m.group(0))
            break
    res.allowed = not res.categories
    return res
