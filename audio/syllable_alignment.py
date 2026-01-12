"""
Syllable-to-Onset Alignment

The problem:
- Whisper gives word-level timestamps ("beautiful" from 1.0s to 1.3s)
- Onset detection gives rhythmic events (1.0s, 1.1s, 1.2s)
- We need to split "beautiful" → "beau", "ti", "ful" across those 3 onsets

Solution:
1. For each word, count how many onsets fall within its timespan
2. Split the word into that many syllables (using pyphen)
3. Assign one syllable to each onset
"""

import pyphen
from dataclasses import dataclass
from typing import Optional


# Initialize hyphenator
_hyphenator = None

def get_hyphenator():
    global _hyphenator
    if _hyphenator is None:
        _hyphenator = pyphen.Pyphen(lang='en_US')
    return _hyphenator


@dataclass
class AlignedSyllable:
    """A syllable aligned to a specific onset time"""
    index: int
    text: str                    # The syllable text ("beau")
    word: str                    # Parent word ("beautiful")
    start_sec: float             # Onset time
    end_sec: float               # End time (next onset or word end)
    bar: int
    beat: float
    is_word_start: bool          # First syllable of word?
    is_word_end: bool            # Last syllable of word?
    confidence: float            # How confident in this alignment


def split_word_into_syllables(word: str) -> list:
    """
    Split a word into syllables using pyphen.

    "beautiful" → ["beau", "ti", "ful"]
    "I" → ["I"]
    "don't" → ["don't"]
    """
    clean = word.strip(".,!?;:'\"")

    if not clean:
        return []

    # Single character or very short
    if len(clean) <= 2:
        return [word]  # Keep punctuation

    hyph = get_hyphenator()

    # pyphen returns "beau-ti-ful"
    hyphenated = hyph.inserted(clean)
    syllables = hyphenated.split('-')

    # If no hyphens found, return as single syllable
    if len(syllables) == 1:
        return [word]

    # Reattach trailing punctuation to last syllable
    trailing = ''
    for char in word[::-1]:
        if char in '.,!?;:\'"':
            trailing = char + trailing
        else:
            break

    if trailing:
        syllables[-1] = syllables[-1] + trailing

    return syllables


def align_words_to_onsets(
    whisper_words: list,
    onset_times: list,
    bar_duration: float,
    tolerance_ms: float = 50
) -> list:
    """
    Align whisper words to detected onsets, splitting words into syllables.

    Args:
        whisper_words: List of {"text": str, "start": float, "end": float}
        onset_times: List of onset times in seconds
        bar_duration: Duration of one bar in seconds
        tolerance_ms: Tolerance for matching (ms)

    Returns:
        List of AlignedSyllable objects
    """
    tolerance = tolerance_ms / 1000
    aligned = []
    syllable_idx = 0

    for word_info in whisper_words:
        word_text = word_info['text'].strip()
        word_start = word_info['start']
        word_end = word_info['end']

        if not word_text:
            continue

        # Find onsets that fall within this word's timespan
        word_onsets = []
        for onset in onset_times:
            if word_start - tolerance <= onset <= word_end + tolerance:
                word_onsets.append(onset)

        # If no onsets in this word, use word start as single onset
        if not word_onsets:
            word_onsets = [word_start]

        # Split word into syllables
        syllables = split_word_into_syllables(word_text)

        # If more onsets than syllables, distribute syllables across onsets
        # If more syllables than onsets, combine syllables

        if len(word_onsets) >= len(syllables):
            # More onsets than syllables: assign syllables to first N onsets
            for i, syl_text in enumerate(syllables):
                onset_time = word_onsets[i] if i < len(word_onsets) else word_onsets[-1]

                # Calculate end time
                if i < len(word_onsets) - 1:
                    end_time = word_onsets[i + 1]
                else:
                    end_time = word_end

                # Calculate bar/beat
                bar_num = int(onset_time / bar_duration) + 1
                beat_in_bar = ((onset_time % bar_duration) / bar_duration) * 4 + 1

                aligned.append(AlignedSyllable(
                    index=syllable_idx,
                    text=syl_text,
                    word=word_text,
                    start_sec=onset_time,
                    end_sec=end_time,
                    bar=bar_num,
                    beat=beat_in_bar,
                    is_word_start=(i == 0),
                    is_word_end=(i == len(syllables) - 1),
                    confidence=0.9 if len(syllables) == len(word_onsets) else 0.7
                ))
                syllable_idx += 1

        else:
            # More syllables than onsets: combine syllables to match onset count
            syls_per_onset = len(syllables) / len(word_onsets)

            for i, onset_time in enumerate(word_onsets):
                # Which syllables go with this onset?
                start_syl = int(i * syls_per_onset)
                end_syl = int((i + 1) * syls_per_onset)
                combined_text = ''.join(syllables[start_syl:end_syl])

                if not combined_text:
                    combined_text = syllables[min(start_syl, len(syllables)-1)]

                # Calculate end time
                if i < len(word_onsets) - 1:
                    end_time = word_onsets[i + 1]
                else:
                    end_time = word_end

                bar_num = int(onset_time / bar_duration) + 1
                beat_in_bar = ((onset_time % bar_duration) / bar_duration) * 4 + 1

                aligned.append(AlignedSyllable(
                    index=syllable_idx,
                    text=combined_text,
                    word=word_text,
                    start_sec=onset_time,
                    end_sec=end_time,
                    bar=bar_num,
                    beat=beat_in_bar,
                    is_word_start=(i == 0),
                    is_word_end=(i == len(word_onsets) - 1),
                    confidence=0.6  # Lower confidence when combining
                ))
                syllable_idx += 1

    return aligned


def align_words_to_onsets_greedy(
    whisper_words: list,
    onset_times: list,
    bar_duration: float
) -> list:
    """
    Greedy alignment: assign each onset to the closest word,
    then split that word's syllables across consecutive onsets.

    This prevents text bleeding across word boundaries.
    """
    if not whisper_words or not onset_times:
        return []

    aligned = []
    onset_idx = 0
    syllable_idx = 0

    for word_info in whisper_words:
        word_text = word_info['text'].strip()
        word_start = word_info['start']
        word_end = word_info['end']

        if not word_text:
            continue

        # Count onsets that belong to this word
        # An onset belongs to this word if it's closer to this word than adjacent words
        word_onset_count = 0
        temp_idx = onset_idx

        while temp_idx < len(onset_times):
            onset = onset_times[temp_idx]

            # Stop if onset is past this word's end (with tolerance)
            if onset > word_end + 0.1:
                break

            # Count if onset is after word start (with tolerance)
            if onset >= word_start - 0.05:
                word_onset_count += 1

            temp_idx += 1

        # Ensure at least one "onset" per word
        if word_onset_count == 0:
            word_onset_count = 1

        # Split word into syllables
        syllables = split_word_into_syllables(word_text)

        # Distribute syllables across onsets
        for i in range(word_onset_count):
            # Which syllable(s) for this onset?
            if word_onset_count >= len(syllables):
                # More onsets than syllables - some onsets get partial/repeated
                syl_idx = min(i, len(syllables) - 1)
                syl_text = syllables[syl_idx]
            else:
                # More syllables than onsets - combine
                syls_per_onset = len(syllables) / word_onset_count
                start_s = int(i * syls_per_onset)
                end_s = int((i + 1) * syls_per_onset)
                syl_text = ''.join(syllables[start_s:end_s]) or syllables[-1]

            # Get onset time
            if onset_idx < len(onset_times):
                onset_time = onset_times[onset_idx]
                onset_idx += 1
            else:
                onset_time = word_start + (i * (word_end - word_start) / word_onset_count)

            # End time
            if onset_idx < len(onset_times):
                end_time = min(onset_times[onset_idx], word_end)
            else:
                end_time = word_end

            bar_num = int(onset_time / bar_duration) + 1
            beat = ((onset_time % bar_duration) / bar_duration) * 4 + 1

            aligned.append(AlignedSyllable(
                index=syllable_idx,
                text=syl_text,
                word=word_text,
                start_sec=onset_time,
                end_sec=end_time,
                bar=bar_num,
                beat=beat,
                is_word_start=(i == 0),
                is_word_end=(i == word_onset_count - 1),
                confidence=0.8
            ))
            syllable_idx += 1

    return aligned


def demo_alignment(word: str, word_start: float, word_end: float, onsets: list):
    """Show how a word gets split across onsets"""

    syllables = split_word_into_syllables(word)

    # Find onsets in word's range
    word_onsets = [o for o in onsets if word_start <= o <= word_end]

    print(f"\nWord: '{word}' ({word_start:.2f}s - {word_end:.2f}s)")
    print(f"  Syllables: {syllables}")
    print(f"  Onsets in range: {[f'{o:.2f}s' for o in word_onsets]}")
    print(f"  Result:")

    if len(word_onsets) >= len(syllables):
        for i, syl in enumerate(syllables):
            onset = word_onsets[i] if i < len(word_onsets) else "?"
            print(f"    {onset} → '{syl}'")
    else:
        syls_per = len(syllables) / max(len(word_onsets), 1)
        for i, onset in enumerate(word_onsets):
            s = int(i * syls_per)
            e = int((i+1) * syls_per)
            combined = ''.join(syllables[s:e])
            print(f"    {onset:.2f}s → '{combined}'")


if __name__ == "__main__":
    # Demo
    print("SYLLABLE SPLITTING DEMO")
    print("=" * 50)

    test_words = ["beautiful", "surprise", "I", "don't", "fuck", "amazing", "everybody"]
    for w in test_words:
        syls = split_word_into_syllables(w)
        print(f"  {w:<15} → {syls}")

    print("\n" + "=" * 50)
    print("ALIGNMENT DEMO")

    # Simulate: "beautiful" from 1.0s to 1.3s with 3 onsets
    demo_alignment("beautiful", 1.0, 1.3, [1.0, 1.1, 1.2, 1.5])

    # Simulate: "I" from 0.5s to 0.6s with 1 onset
    demo_alignment("I", 0.5, 0.6, [0.5, 0.7])

    # Simulate: "everybody" from 2.0s to 2.5s with 2 onsets (fewer than syllables)
    demo_alignment("everybody", 2.0, 2.5, [2.0, 2.25, 3.0])
