"""Deterministic, field-aware text representations for entity resolution."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache


# These are deliberately small, auditable maps. They are preprocessing rules,
# not the mechanism used to decide whether two businesses are the same.
LEGAL_SUFFIXES = frozenset({
    "corp", "corporation", "inc", "incorporated", "llc", "ltd", "limited",
    "pvt", "private", "pte", "company", "co", "sarl", "srl", "sa", "sas",
    "eirl", "plc", "llp", "lp",
})
LEGAL_SUFFIX_PHRASES = (
    ("limited", "liability", "company"),
    ("private", "limited"),
    ("public", "limited"),
)

NAME_REPLACEMENTS = (
    (re.compile(r"\bcorp\b"), "corporation"),
    (re.compile(r"\binc\b"), "incorporated"),
    (re.compile(r"\bllc\b"), "limited liability company"),
    (re.compile(r"\bltd\b"), "limited"),
    (re.compile(r"\bpvt\b"), "private"),
    (re.compile(r"\bpte\b"), "private"),
    (re.compile(r"\bco\b"), "company"),
    (re.compile(r"\bintl\b"), "international"),
    (re.compile(r"\bmgmt\b"), "management"),
    (re.compile(r"\bsvc\b"), "service"),
    (re.compile(r"\bsvcs\b"), "services"),
    (re.compile(r"\bmfg\b"), "manufacturing"),
    (re.compile(r"\bdist\b"), "distribution"),
    (re.compile(r"\bgrp\b"), "group"),
    (re.compile(r"\bassoc\b"), "associates"),
    (re.compile(r"\bsoln\b"), "solution"),
    (re.compile(r"\bsolns\b"), "solutions"),
    (re.compile(r"\btech\b"), "technology"),
    (re.compile(r"\btechs\b"), "technologies"),
)

ADDRESS_REPLACEMENTS = (
    (re.compile(r"\brd\b"), "road"), (re.compile(r"\bst\b"), "street"),
    (re.compile(r"\bave?\b"), "avenue"), (re.compile(r"\bblvd\b"), "boulevard"),
    (re.compile(r"\bdr\b"), "drive"), (re.compile(r"\bln\b"), "lane"),
    (re.compile(r"\bct\b"), "court"), (re.compile(r"\bpl\b"), "place"),
    (re.compile(r"\bsq\b"), "square"), (re.compile(r"\bpkwy\b"), "parkway"),
    (re.compile(r"\bhwy\b"), "highway"), (re.compile(r"\bapt\b|\bappt\b"), "apartment"),
    (re.compile(r"\bste\b"), "suite"), (re.compile(r"\bflr?\b"), "floor"),
    (re.compile(r"\bbldg\b"), "building"), (re.compile(r"\bno\b"), "number"),
)

_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s\-']", flags=re.UNICODE)
_NUMBERS = re.compile(r"\d+")
_ALNUM = re.compile(r"[^\w]+", flags=re.UNICODE)
_STOP_TOKENS = frozenset({
    "the", "and", "of", "for", "company", "corporation", "incorporated",
    "limited", "private", "service", "services", "group", "road", "street",
    "avenue", "building", "india", "france", "usa", "united", "states",
})


def _unicode_safe(text: str) -> str:
    """Normalise Unicode without deleting predominantly non-Latin text."""
    text = unicodedata.normalize("NFKC", text)
    ascii_text = text.encode("ascii", "ignore").decode("ascii").strip()
    if text.strip() and len(ascii_text) >= len(text.strip()) * 0.5:
        return ascii_text
    return "".join(
        char for char in text
        if unicodedata.category(char) not in {"Cs", "Cc", "Co"}
    )


@lru_cache(maxsize=500_000)
def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _unicode_safe(text).lower()
    text = re.sub(r"\s*&\s*", " and ", text)
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


@lru_cache(maxsize=500_000)
def normalize_business_name(name: str) -> str:
    text = clean_text(name)
    for pattern, replacement in NAME_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return _SPACE.sub(" ", text).strip()


@lru_cache(maxsize=500_000)
def normalize_address(address: str) -> str:
    text = clean_text(address)
    for pattern, replacement in ADDRESS_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return _SPACE.sub(" ", text).strip()


def name_core(name_normalized: str) -> str:
    """Remove only trailing legal-form tokens; retain the original form separately."""
    tokens = name_normalized.split()
    changed = True
    while tokens and changed:
        changed = False
        for phrase in LEGAL_SUFFIX_PHRASES:
            if len(tokens) >= len(phrase) and tuple(tokens[-len(phrase):]) == phrase:
                del tokens[-len(phrase):]
                changed = True
                break
        if not changed and tokens and tokens[-1] in LEGAL_SUFFIXES:
            tokens.pop()
            changed = True
    return " ".join(tokens)

def legal_type(name_normalized: str) -> str:
    core = name_core(name_normalized)
    return name_normalized[len(core):].strip() if core else name_normalized


def compact_text(text: str) -> str:
    return _ALNUM.sub("", text)


def sorted_tokens(text: str) -> str:
    return " ".join(sorted(text.split()))


def initials(text: str) -> str:
    tokens = [token for token in text.split() if token not in _STOP_TOKENS and token]
    if len(tokens) == 1 and 2 <= len(tokens[0]) <= 8:
        return tokens[0]
    return "".join(token[0] for token in tokens)


def numeric_tokens(text: str) -> tuple[str, ...]:
    return tuple(_NUMBERS.findall(text))


def significant_tokens(text: str) -> tuple[str, ...]:
    tokens = [
        token for token in text.split()
        if len(token) >= 4 and not token.isdigit() and token not in _STOP_TOKENS
    ]
    return tuple(sorted(set(tokens), key=lambda token: (-len(token), token)))


def normalize_country(country: str) -> str:
    return clean_text(country)


def prepare_record(name: str, address: str, country: str) -> dict[str, object]:
    """Return compact representations used by retrieval and pair features."""
    raw_name = clean_text(name)
    raw_address = clean_text(address)
    norm_name = normalize_business_name(name)
    norm_address = normalize_address(address)
    core_name = name_core(norm_name)
    return {
        "name_raw": raw_name,
        "addr_raw": raw_address,
        "name_norm": norm_name,
        "name_core": core_name,
        "name_compact": compact_text(core_name),
        "name_sorted": sorted_tokens(core_name),
        "name_initials": initials(core_name),
        "legal_type": legal_type(norm_name),
        "addr_norm": norm_address,
        "addr_compact": compact_text(norm_address),
        "addr_sorted": sorted_tokens(norm_address),
        "addr_numbers": numeric_tokens(norm_address),
        "name_tokens": significant_tokens(core_name),
        "addr_tokens": significant_tokens(norm_address),
        "country": normalize_country(country),
    }


def make_combined_text(row) -> str:
    prepared = prepare_record(
        str(row.get("business_name", "") or ""),
        str(row.get("business_address", "") or ""),
        str(row.get("country", "") or ""),
    )
    return f"{prepared['name_core']} {prepared['name_core']} {prepared['addr_norm']}".strip()
