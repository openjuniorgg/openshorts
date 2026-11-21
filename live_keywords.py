"""Keyword gate for live speech and chat (English and Portuguese).

This module only decides WHEN a live moment is worth sending to the clip
pipeline. It never rates anything: a hit is a fact (a word was said), not a
score. the AI model rates the moment (predicted_score) and decides the clip.

Text is normalized before matching: lower case, accents removed, curly
apostrophes removed, so "vitória", "Vitoria" and "VITÓRIA" are one word.

The game gate is GTA VI only. Streams whose title or category point to GTA V,
GTA Online or Red Dead Redemption 2 are not monitored.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List

# Clip requests. Strong: one hit is enough to ask the AI model.
CLIP_REQUEST = [
    "clip", "clip that", "clip it", "clip this", "clipa", "clipar", "clipa isso", "clipa essa",
    "clipa aquilo", "clipar isso", "clipe", "faz o clipe", "cria o clipe", "corta isso",
    "cortar isso", "clip now",
]

# Screams and shouts, usually with no clear words. Strong.
SCREAM = [
    "ahh", "ahhh", "aaah", "aaaa", "aaaaa", "ai meu deus", "ai caramba", "socorro",
    "watch out", "nao nao", "para para", "no no no", "holy shit",
]

# Laughter in chat and voice. Weak: one hit is not enough, two in a chunk are.
LAUGHTER = [
    "haha", "hahaha", "hahahaha", "hehe", "hehehe", "kkk", "kkkk", "kkkkk", "rsrs", "rsrsrs",
    "jaja", "jajaja", "jajajaja", "lol", "lmao", "rofl", "xd", "mdr", "morri de rir",
    "to morrendo", "estou morrendo", "dying", "im dying", "i'm dying", "dead",
]

# Reactions: surprise, hype, agreement, disbelief. Weak.
REACTION = [
    "no way", "omg", "oh my god", "wtf", "insane", "crazy", "lets go", "let's go",
    "clutch", "gg", "wow", "bruh", "brooo", "nossa", "caraca", "que isso", "nao acredito",
    "incrivel", "mano", "bora", "dios mio", "madre mia", "que loucura", "que locura",
    "nao e possivel", "impossivel", "sem chance", "absurdo", "pqp", "puta merda",
    "meu deus", "mds", "nope", "caralho", "porra", "puta que pariu", "eu nao acredito",
]

# Events that happen in GTA VI gameplay (EN + PT). Weak.
GAME_EVENT = [
    "police", "cops", "cop", "policia", "wanted", "wanted level", "estrela", "estrelas",
    "chase", "perseguicao", "fuga", "fugi", "escapei", "heist", "assalto", "roubo",
    "mission", "missao", "missoes", "explosion", "explodiu", "explodir", "boom", "explode",
    "crash", "bug", "bugs", "bugado", "glitch", "glitched", "npc", "npcs", "troll",
    "atropelei", "atropelou", "atropelar", "ran over", "run over", "bati", "batida",
    "tiro", "tiros", "shot", "headshot", "kill", "killed", "matei", "abati", "abate",
    "morri", "died", "death", "dead", "ace", "victory", "vitoria", "ganhei", "perdi",
    "derrota", "boss", "chefe", "helicopter", "helicoptero", "tank", "tanque", "jet",
    "aviao", "lucia", "jason", "vice city", "leonida", "trailer", "sniper", "rpg",
    "nuke", "nuclear", "parachute", "paraquedas", "corre", "foge", "cuidado", "sai daqui",
]

CATEGORIES: Dict[str, List[str]] = {
    "clip_request": CLIP_REQUEST,
    "scream": SCREAM,
    "laughter": LAUGHTER,
    "reaction": REACTION,
    "game_event": GAME_EVENT,
}

# A single hit in one of these is enough to make a candidate.
STRONG_CATEGORIES = {"clip_request", "scream"}

# Laughter written with repeated letters: hahahaha, kkkkkkkk, rsrsrsrs, jajajaja.
_PATTERN_EXTRA: Dict[str, List[str]] = {
    "laughter": [
        r"\b(?:ha){2,}h?\b", r"\b(?:he){2,}h?\b", r"\bk{2,}\b", r"\bja(?:ja)+j?a?\b",
        r"\b(?:rs){2,}\b", r"\bl+o+l+\b", r"\bx+d+\b",
    ],
    "scream": [r"\ba{3,}h*\b", r"\bah{2,}\b"],
}


def normalize(text: str) -> str:
    """Lower case, no accents, no curly apostrophes, single spaces."""
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def _term_pattern(term: str) -> str:
    words = [re.escape(w) for w in normalize(term).split(" ") if w]
    return r"\b" + r"\s+".join(words) + r"\b"


def _compile(category: str) -> re.Pattern:
    parts = [_term_pattern(t) for t in CATEGORIES[category]]
    parts += _PATTERN_EXTRA.get(category, [])
    return re.compile("|".join(f"(?:{p})" for p in parts))


_PATTERNS = {cat: _compile(cat) for cat in CATEGORIES}


def find_hits(text: str) -> Dict[str, List[str]]:
    """The keyword hits in ``text``, by category (unique, in order of first match).

    Only categories with at least one hit are returned.
    """
    norm = normalize(text)
    if not norm:
        return {}
    hits: Dict[str, List[str]] = {}
    for category, pattern in _PATTERNS.items():
        found: List[str] = []
        for match in pattern.finditer(norm):
            word = match.group(0)
            if word not in found:
                found.append(word)
        if found:
            hits[category] = found
    return hits


def is_candidate(hits: Dict[str, List[str]]) -> bool:
    """A moment is worth sending to the pipeline: a strong hit, or two weak ones."""
    if any(hits.get(cat) for cat in STRONG_CATEGORIES):
        return True
    return sum(len(v) for v in hits.values()) >= 2


# --- Game gate: GTA VI only ----------------------------------------------- #

# Explicit GTA VI names. "vice city" and "leonida" alone are not enough: they
# also name places in older GTA games.
_GTA6_RE = re.compile(
    r"\bgta\s*(?:6|vi|six|seis)\b|\bgrand theft auto\s*(?:6|vi|six|seis)\b"
)
# Other games that must never pass the gate.
_OTHER_GAME_RE = re.compile(
    r"\bgta\s*(?:v|5|iv|4|iii|3|sa|vc|online)\b|\bgrand theft auto\s*(?:v|5|iv|4|iii|3)\b"
    r"|\bgta\s+(?:san andreas|vice city)\b|\bred dead\b|\brdr\s*2\b|\brdr2\b"
)


def is_gta6_stream(title: str, category: str) -> bool:
    """True when the stream's category or title says GTA VI.

    The category decides when it is set. Otherwise the title must name GTA VI
    and must not name another game in the GTA series or Red Dead.
    """
    cat = normalize(category)
    ttl = normalize(title)
    if cat and _GTA6_RE.search(cat):
        return not _OTHER_GAME_RE.search(cat)
    if cat and _OTHER_GAME_RE.search(cat):
        return False
    if _OTHER_GAME_RE.search(ttl):
        return False
    return bool(_GTA6_RE.search(ttl))


def hit_words(hits: Dict[str, List[str]]) -> Iterable[str]:
    """Every matched word, flattened (for logs and reasons)."""
    for words in hits.values():
        for word in words:
            yield word
