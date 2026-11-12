"""
Planted errors: deterministic corruptions of a correct draft, used to measure
whether the verifier catches real mistakes (catch rate) and whether it flags
drafts that are fine (false-alarm rate). No LLM is used to corrupt, so every
corruption is reproducible and can be inspected before any API money is spent.
"""

import re
from dataclasses import dataclass

# well-known directors used as swap targets; a swap only happens when the
# replacement name does not appear anywhere in the evidence
DIRECTOR_POOL = (
    "Christopher Nolan", "Steven Spielberg", "James Cameron", "Martin Scorsese",
    "Ridley Scott", "Quentin Tarantino", "Peter Jackson", "Tim Burton",
    "David Fincher", "Clint Eastwood", "Woody Allen", "Ron Howard",
)
# plausible-sounding but absent from the corpus (it holds no award data)
UNSUPPORTED_CLAIMS = (
    "The film also won several major industry awards.",
    "A sequel to the film was announced shortly after its release.",
    "The film was shot almost entirely on location abroad.",
)
CORRUPTION_TYPES = ("swap_entity", "change_number", "add_unsupported_claim")

_NUMBER = re.compile(r"(?<![\w.$])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w])")


@dataclass
class Corruption:
    type: str
    draft: str
    detail: str


def swap_entity(draft: str, evidence_text: str) -> Corruption | None:
    """Replace one director named in the draft with a different one the evidence never mentions."""
    present = [d for d in DIRECTOR_POOL if d.lower() in draft.lower()]
    if not present:
        return None
    old = present[0]
    new = next((d for d in DIRECTOR_POOL if d != old and d.lower() not in evidence_text.lower()), None)
    if new is None:
        return None
    return Corruption("swap_entity", re.sub(re.escape(old), new, draft, flags=re.I),
                      f"{old} -> {new}")


def _shift(token: str) -> str:
    """Change a number plausibly: years +3, ratings +1.5 (capped), other integers x2."""
    plain = token.replace(",", "")
    if "." in plain:
        v = float(plain)
        return f"{v - 1.5 if v > 8.5 else v + 1.5:.1f}"
    v = int(plain)
    if 1900 <= v <= 2100:
        return str(v + 3)
    return f"{v * 2:,}" if "," in token else str(v * 2)


def change_number(draft: str) -> Corruption | None:
    """Alter the first meaningful number in the draft (skips list markers like '1.')."""
    for m in _NUMBER.finditer(draft):
        tok = m.group(1)
        at_line_start = draft[: m.start()].rstrip(" ").endswith("\n") or m.start() == 0
        if at_line_start and draft[m.end(): m.end() + 1] in (".", ")"):
            continue  # numbered list marker, not data
        new = _shift(tok)
        if new == tok:
            continue
        return Corruption("change_number", draft[: m.start(1)] + new + draft[m.end(1):],
                          f"{tok} -> {new}")
    return None


def add_unsupported_claim(draft: str, index: int = 0) -> Corruption | None:
    """Append a plausible claim that nothing in the evidence supports."""
    claim = UNSUPPORTED_CLAIMS[index % len(UNSUPPORTED_CLAIMS)]
    return Corruption("add_unsupported_claim", f"{draft.rstrip()}\n\n{claim}", f"appended: {claim}")


def corrupt_all(draft: str, evidence_text: str, index: int = 0) -> list[Corruption]:
    """Every corruption that applies to this draft (types that don't apply are skipped)."""
    out = [swap_entity(draft, evidence_text), change_number(draft), add_unsupported_claim(draft, index)]
    return [c for c in out if c is not None and c.draft != draft]


# -- harder, subtler corruptions ---------------------------------------------
# The easy set above contradicts the evidence outright. These stay close to the
# truth: a number nudged slightly, an item silently dropped from a list, two
# years traded between movies.

HARD_TYPES = ("near_miss_number", "drop_list_item", "swap_years")
_YEAR = re.compile(r"\b(19\d\d|20\d\d)\b")
_LIST_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")


def near_miss_number(draft: str) -> Corruption | None:
    """Nudge a number slightly: year +1, rating +0.1, large figures +3%, small counts +1."""
    for m in _NUMBER.finditer(draft):
        tok = m.group(1)
        at_line_start = draft[: m.start()].rstrip(" ").endswith("\n") or m.start() == 0
        if at_line_start and draft[m.end(): m.end() + 1] in (".", ")"):
            continue  # list marker
        plain = tok.replace(",", "")
        if "." in plain:
            new = f"{float(plain) + 0.1:.1f}"
        else:
            v = int(plain)
            if 1900 <= v <= 2100:
                new = str(v + 1)
            elif v >= 1000:
                new = f"{round(v * 1.03):,}" if "," in tok else str(round(v * 1.03))
            else:
                new = str(v + 1)
        if new != tok:
            return Corruption("near_miss_number", draft[: m.start(1)] + new + draft[m.end(1):],
                              f"{tok} -> {new}")
    return None


def drop_list_item(draft: str) -> Corruption | None:
    """Remove the middle line of a list of 3+ items (an omission, not a contradiction)."""
    lines = draft.split("\n")
    idx = [i for i, ln in enumerate(lines) if _LIST_LINE.match(ln)]
    if len(idx) < 3:
        return None
    drop = idx[len(idx) // 2]
    removed = lines[drop].strip()
    return Corruption("drop_list_item", "\n".join(lines[:drop] + lines[drop + 1:]),
                      f"dropped: {removed[:80]}")


def swap_years(draft: str) -> Corruption | None:
    """Trade the first two different years in the draft (each film gets the other's year)."""
    found = [(m.start(1), m.end(1), m.group(1)) for m in _YEAR.finditer(draft)]
    if not found:
        return None
    first = found[0]
    second = next((f for f in found[1:] if f[2] != first[2]), None)
    if second is None:
        return None
    (s1, e1, y1), (s2, e2, y2) = first, second
    return Corruption("swap_years", draft[:s1] + y2 + draft[e1:s2] + y1 + draft[e2:], f"{y1} <-> {y2}")


def corrupt_hard(draft: str) -> list[Corruption]:
    """Every hard corruption that applies to this draft."""
    out = [near_miss_number(draft), drop_list_item(draft), swap_years(draft)]
    return [c for c in out if c is not None and c.draft != draft]
