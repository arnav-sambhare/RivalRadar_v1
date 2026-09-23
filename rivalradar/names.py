"""Organisation-name normalisation and matching.

Patent offices and bibliographic databases spell the same company many ways:
"RIVAL ONE INC [US]", "Rival One, Inc.", "Rival One GmbH". Matching on the raw
string misses these; matching on loose substrings over-matches. We normalise
(lowercase, strip punctuation, country tags, and legal suffixes) and then
require the rival's normalised tokens to be a prefix of the candidate's.
"""

from __future__ import annotations

import re

# Legal-form suffixes stripped from the end of a name. Kept to common forms;
# extend as real data turns up misses.
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd",
    "limited", "llc", "llp", "plc", "gmbh", "ag", "sa", "sas", "sarl", "srl",
    "spa", "bv", "nv", "ab", "as", "oy", "kk", "pty", "pte", "ug",
}

_COUNTRY_TAG = re.compile(r"\[[a-z]{2}\]")
_PARENS = re.compile(r"\([^)]*\)")  # OpenAlex tags, e.g. "Amgen (United States)"
_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize_org(name: str) -> str:
    """Lowercase, drop country tags, parentheses and punctuation, strip trailing
    legal suffixes."""
    text = _PARENS.sub(" ", _COUNTRY_TAG.sub(" ", (name or "").lower()))
    tokens = _NON_WORD.sub(" ", text).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def org_matches(rival_name: str, candidate: str) -> bool:
    """True if the candidate names the same organisation as the rival.

    The rival's normalised tokens must be a prefix of the candidate's, so
    "Mistral AI" matches "MISTRAL AI FRANCE SAS" but not "Mistral Aerospace".
    """
    want = normalize_org(rival_name).split()
    have = normalize_org(candidate).split()
    return bool(want) and have[: len(want)] == want
