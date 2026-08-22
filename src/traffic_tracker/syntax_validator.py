"""
syntax_validator.py
-------------------
Production regional syntax validation, slot-aware OCR character disambiguation,
and candidate re-scoring for commercial-grade ALPR/ANPR.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# -- Alphanumeric Confusion Lookup Tables ----------------------------------
# Standard bidirectional character confusions in OCR
DIGIT_MAP: Dict[str, str] = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1", "T": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6", "C": "6",
    "B": "8",
    "J": "9",
}

LETTER_MAP: Dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
    "9": "J",
}

# -- Regional Template Definitions ------------------------------------------
# Schema: (slot_pattern_types, regex_validator, jurisdiction_code, priority)
# slot_pattern_types: list where 'D' = digit required, 'L' = letter required, 'A' = alphanumeric
TEMPLATES = [
    # US California Standard: 1 Digit - 3 Letters - 3 Digits (e.g., 9JWM255, 8MZW276, 9GAD429)
    {
        "slots": ["D", "L", "L", "L", "D", "D", "D"],
        "regex": re.compile(r"^[0-9][A-Z]{3}[0-9]{3}$"),
        "jurisdiction": "us-ca",
        "len": 7,
        "weight": 1.6,
    },
    # US California Commercial / Truck: 1 Digit - 2 Letters - 4 Digits (e.g., 5A12345)
    {
        "slots": ["D", "L", "L", "D", "D", "D", "D"],
        "regex": re.compile(r"^[0-9][A-Z]{2}[0-9]{4}$"),
        "jurisdiction": "us-ca-comm",
        "len": 7,
        "weight": 1.4,
    },
    # US General (Texas / New York / Florida): 3 Letters - 4 Digits (e.g., ABC1234)
    {
        "slots": ["L", "L", "L", "D", "D", "D", "D"],
        "regex": re.compile(r"^[A-Z]{3}[0-9]{4}$"),
        "jurisdiction": "us-gen",
        "len": 7,
        "weight": 1.5,
    },
    # US General: 3 Digits - 3 Letters (e.g., 123ABC)
    {
        "slots": ["D", "D", "D", "L", "L", "L"],
        "regex": re.compile(r"^[0-9]{3}[A-Z]{3}$"),
        "jurisdiction": "us-gen",
        "len": 6,
        "weight": 1.4,
    },
    # US General: 3 Letters - 3 Digits (e.g., ABC123)
    {
        "slots": ["L", "L", "L", "D", "D", "D"],
        "regex": re.compile(r"^[A-Z]{3}[0-9]{3}$"),
        "jurisdiction": "us-gen",
        "len": 6,
        "weight": 1.4,
    },
    # UK Standard (e.g., BD51SMR): 2 Letters - 2 Digits - 3 Letters
    {
        "slots": ["L", "L", "D", "D", "L", "L", "L"],
        "regex": re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$"),
        "jurisdiction": "gb",
        "len": 7,
        "weight": 1.5,
    },
    # EU Standard (France / Italy): 2 Letters - 3 Digits - 2 Letters (e.g., AA123AA)
    {
        "slots": ["L", "L", "D", "D", "D", "L", "L"],
        "regex": re.compile(r"^[A-Z]{2}[0-9]{3}[A-Z]{2}$"),
        "jurisdiction": "eu",
        "len": 7,
        "weight": 1.5,
    },
]


class RegionalSyntaxValidator:
    """
    Validates, corrects, and rescores OCR character strings against regional syntax rules.
    """

    def __init__(self, default_jurisdiction: str = "us-ca"):
        self.default_jurisdiction = default_jurisdiction

    @staticmethod
    def clean_raw_string(text: str) -> str:
        """Strip punctuation, spaces, accents, and normalize to uppercase alphanumeric."""
        if not text:
            return ""
        return re.sub(r"[^0-9A-Z]", "", text.upper())

    def validate_and_correct(
        self,
        raw_text: str,
        preferred_jurisdiction: Optional[str] = None,
    ) -> Tuple[str, bool, str, float]:
        """
        Disambiguates OCR text and checks conformity against active regional templates.
        
        Args:
            raw_text: Raw or decoded plate string.
            preferred_jurisdiction: 'us-ca', 'us-gen', 'gb', 'eu', or 'auto'.
            
        Returns:
            (corrected_text, is_valid, jurisdiction, confidence_multiplier)
        """
        clean = self.clean_raw_string(raw_text)
        if len(clean) < 3 or len(set(clean)) <= 2:
            return "", False, "unknown", 0.0

        n = len(clean)
        pref = preferred_jurisdiction or self.default_jurisdiction

        best_cand = clean
        best_valid = False
        best_jur = "generic"
        best_weight = 1.0

        for tmpl in TEMPLATES:
            # Check length match
            if tmpl["len"] != n:
                continue

            is_preferred = (pref == "auto" or tmpl["jurisdiction"].startswith(pref) or pref.startswith(tmpl["jurisdiction"]))

            chars = list(clean)
            corrected_chars = []
            slot_mismatches = 0

            for ch, slot_type in zip(chars, tmpl["slots"]):
                if slot_type == "D":
                    if ch.isdigit():
                        corrected_chars.append(ch)
                    elif ch in DIGIT_MAP:
                        corrected_chars.append(DIGIT_MAP[ch])
                    else:
                        corrected_chars.append(ch)
                        slot_mismatches += 1
                elif slot_type == "L":
                    if ch.isalpha():
                        corrected_chars.append(ch)
                    elif ch in LETTER_MAP:
                        corrected_chars.append(LETTER_MAP[ch])
                    else:
                        corrected_chars.append(ch)
                        slot_mismatches += 1
                else:
                    corrected_chars.append(ch)

            cand_str = "".join(corrected_chars)
            if tmpl["regex"].match(cand_str) and len(set(cand_str)) > 2:
                w = tmpl["weight"] * (1.2 if is_preferred else 1.0)
                if w > best_weight or not best_valid:
                    best_cand = cand_str
                    best_valid = True
                    best_jur = tmpl["jurisdiction"]
                    best_weight = w
                    if is_preferred:
                        break

        return best_cand, best_valid, best_jur, best_weight

    def rescore_candidates(
        self,
        candidates: List[Tuple[str, float]],
        preferred_jurisdiction: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float, str, bool]]:
        """
        Takes raw CTC beam search candidates [(text, raw_conf), ...] and returns
        rescore ranked candidates [(corrected_text, score, jurisdiction, is_valid), ...].
        """
        rescored = []
        seen = set()

        for raw_text, conf in candidates:
            corr_text, is_valid, jur, weight = self.validate_and_correct(
                raw_text, preferred_jurisdiction=preferred_jurisdiction
            )
            if not corr_text or corr_text in seen:
                continue

            seen.add(corr_text)
            final_score = min(1.0, conf * weight)
            rescored.append((corr_text, round(float(final_score), 3), jur, is_valid))

        # Sort by final score descending
        rescored.sort(key=lambda x: (-x[1], -int(x[3])))
        return rescored[:top_k]


# Module-level singleton instance
default_syntax_validator = RegionalSyntaxValidator(default_jurisdiction="us-ca")
