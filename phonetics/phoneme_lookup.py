"""
phoneme_lookup.py - Robust phoneme lookup with G2P fallback

Handles:
- Standard words (CMU dict)
- Slang, names, ad-libs (G2P neural model)
- Rap-specific normalizations
"""

import re
from functools import lru_cache
from typing import Optional

try:
    import pronouncing
except ImportError:
    pronouncing = None


# Lazy load G2P (it's slow to initialize)
_g2p = None

def get_g2p():
    global _g2p
    if _g2p is None:
        try:
            from g2p_en import G2p
            _g2p = G2p()
        except ImportError:
            _g2p = False  # Mark as unavailable
    return _g2p if _g2p else None


# =============================================================================
# RAP-SPECIFIC NORMALIZATIONS
# =============================================================================

SLANG_NORMALIZATIONS = {
    # Dropping final g
    r"in'$": "ing",
    r"an'$": "and",
    r"em$": "them",
    r"'em$": "them",

    # Common contractions
    r"^ima$": "i'm going to",
    r"^imma$": "i'm going to",
    r"^finna$": "fixing to",
    r"^gonna$": "going to",
    r"^wanna$": "want to",
    r"^gotta$": "got to",
    r"^kinda$": "kind of",
    r"^dunno$": "don't know",
    r"^gimme$": "give me",
    r"^lemme$": "let me",
    r"^tryna$": "trying to",
    r"^boutta$": "about to",
    r"^outta$": "out of",
    r"^lotta$": "lot of",

    # Regional/AAVE
    r"^ain't$": "ain't",
    r"^y'all$": "you all",
    r"^'cause$": "because",
    r"^cuz$": "because",
    r"^wit$": "with",
    r"^dat$": "that",
    r"^dem$": "them",
    r"^dey$": "they",
    r"^dis$": "this",

    # Numbers/slang
    r"^2$": "to",
    r"^4$": "for",
    r"^u$": "you",
    r"^r$": "are",
    r"^n$": "and",
    r"^b$": "be",
}

# Words where we want specific pronunciations
CUSTOM_PHONEMES = {
    # Ad-libs
    "skrrt": [["S", "K", "R", "T"]],
    "brrt": [["B", "R", "T"]],
    "grrt": [["G", "R", "T"]],
    "ayy": [["EY1"]],
    "aye": [["AY1"]],
    "yuh": [["Y", "AH1"]],
    "yeah": [["Y", "EH1"]],
    "nah": [["N", "AA1"]],
    "uh": [["AH1"]],
    "huh": [["HH", "AH1"]],

    # Common slang with known pronunciations
    "drip": [["D", "R", "IH1", "P"]],
    "lit": [["L", "IH1", "T"]],
    "cap": [["K", "AE1", "P"]],
    "bussin": [["B", "AH1", "S", "IH0", "N"]],
    "fye": [["F", "AY1"]],
    "guap": [["G", "W", "AA1", "P"]],
    "hunnid": [["HH", "AH1", "N", "IH0", "D"]],

    # Names (add as needed)
    "drizzy": [["D", "R", "IH1", "Z", "IY0"]],
    "yeezy": [["Y", "IY1", "Z", "IY0"]],
    "hova": [["HH", "OW1", "V", "AH0"]],
}


# =============================================================================
# MAIN LOOKUP
# =============================================================================

@lru_cache(maxsize=10000)
def get_phonemes(word: str) -> list:
    """
    Get phoneme sequences for a word.

    Returns list of possible pronunciations, each a list of ARPAbet phonemes.
    Uses CMU dict first, falls back to G2P for unknown words.

    Example:
        get_phonemes("cat") -> [["K", "AE1", "T"]]
        get_phonemes("read") -> [["R", "IY1", "D"], ["R", "EH1", "D"]]
        get_phonemes("skrrt") -> [["S", "K", "R", "T"]]
    """

    # Clean the word
    word_clean = word.lower().strip()
    word_clean = re.sub(r"[^\w']", "", word_clean)

    if not word_clean:
        return []

    # 1. Check custom dictionary first
    if word_clean in CUSTOM_PHONEMES:
        return CUSTOM_PHONEMES[word_clean]

    # 2. Try CMU dict
    if pronouncing:
        cmu_result = pronouncing.phones_for_word(word_clean)
        if cmu_result:
            return [phones.split() for phones in cmu_result]

    # 3. Try normalized version
    normalized = normalize_slang(word_clean)
    if normalized != word_clean and pronouncing:
        cmu_result = pronouncing.phones_for_word(normalized)
        if cmu_result:
            return [phones.split() for phones in cmu_result]

        # If normalization expanded to multiple words, try each
        if ' ' in normalized:
            combined = []
            for w in normalized.split():
                phones = pronouncing.phones_for_word(w)
                if phones:
                    combined.extend(phones[0].split())
            if combined:
                return [combined]

    # 4. Fall back to G2P
    g2p = get_g2p()
    if g2p:
        g2p_result = g2p(word_clean)

        # G2P returns list with spaces/punctuation, filter to just phonemes
        phonemes = [p for p in g2p_result if p.strip() and p[0].isalpha()]

        if phonemes:
            return [phonemes]

    return []


def normalize_slang(word: str) -> str:
    """Apply rap-specific normalizations"""
    result = word.lower()

    for pattern, replacement in SLANG_NORMALIZATIONS.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


# =============================================================================
# RHYME-SPECIFIC HELPERS
# =============================================================================

# ARPAbet vowels
VOWELS = {
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER',
    'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'
}


def get_rime(phonemes: list) -> tuple:
    """
    Extract the 'rime' (rhyming part) from phonemes.

    Returns (nucleus_vowel, coda_consonants)
    """
    if not phonemes:
        return (None, [])

    # Find the last vowel
    vowel_idx = None
    for i in range(len(phonemes) - 1, -1, -1):
        base = phonemes[i].rstrip('012')
        if base in VOWELS:
            vowel_idx = i
            break

    if vowel_idx is None:
        return (None, phonemes)

    nucleus = phonemes[vowel_idx].rstrip('012')
    coda = phonemes[vowel_idx + 1:]

    return (nucleus, coda)


def get_stressed_vowel(phonemes: list) -> Optional[str]:
    """Get the primary stressed vowel (marked with '1')"""
    for p in phonemes:
        if p.endswith('1'):
            return p.rstrip('1')
    # Fallback to any vowel
    for p in phonemes:
        base = p.rstrip('012')
        if base in VOWELS:
            return base
    return None


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def phonemize_lyrics(words: list) -> list:
    """
    Process a list of words into phoneme data.

    Returns list of dicts with phoneme info for each word.
    """
    results = []

    for word in words:
        phoneme_variants = get_phonemes(word)

        if phoneme_variants:
            primary = phoneme_variants[0]
            nucleus, coda = get_rime(primary)
            stressed = get_stressed_vowel(primary)

            results.append({
                "word": word,
                "phonemes": primary,
                "all_variants": phoneme_variants,
                "nucleus": nucleus,
                "coda": coda,
                "stressed_vowel": stressed,
                "source": "cmu" if pronouncing and pronouncing.phones_for_word(word.lower()) else "g2p"
            })
        else:
            results.append({
                "word": word,
                "phonemes": [],
                "all_variants": [],
                "nucleus": None,
                "coda": [],
                "stressed_vowel": None,
                "source": "failed"
            })

    return results


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    test_words = [
        # Standard words
        "cat", "hat", "running",
        # Slang
        "bussin", "finna", "tryna", "drip",
        # Dropped g
        "runnin'", "gunnin'",
        # Ad-libs
        "skrrt", "brrt", "ayy", "yuh",
        # Names
        "drizzy", "yeezy",
        # Made up
        "flurpington", "zazzle",
    ]

    print(f"{'WORD':<15} {'PHONEMES':<30} {'SOURCE':<10}")
    print("-" * 55)

    for word in test_words:
        phones = get_phonemes(word)
        phone_str = " ".join(phones[0]) if phones else "(none)"

        # Determine source
        if word.lower() in CUSTOM_PHONEMES:
            source = "custom"
        elif pronouncing and pronouncing.phones_for_word(word.lower()):
            source = "cmu"
        else:
            source = "g2p"

        print(f"{word:<15} {phone_str:<30} {source:<10}")
