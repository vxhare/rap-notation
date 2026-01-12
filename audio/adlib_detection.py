"""
Adlib Detection Module

Detects and separates adlibs from main vocals based on:
- Stereo positioning (adlibs often panned hard L/R)
- Timing gaps (adlibs fill spaces between main vocals)
- Energy/loudness differences
- Duration (adlibs are typically short)
- Common adlib vocabulary
"""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Tuple, Optional


# Common adlib words/sounds
ADLIB_VOCABULARY = {
    # Affirmations
    'yeah', 'yuh', 'yah', 'ya', 'uh', 'ah', 'oh', 'ay', 'aye', 'ayy',
    # Exclamations
    'what', 'woo', 'wow', 'damn', 'sheesh', 'gang', 'facts',
    # Ad-lib sounds
    'skrrt', 'skrt', 'brr', 'brrt', 'grr', 'pew', 'bow', 'bang',
    'drip', 'ice', 'cash', 'money', 'lets go', "let's go",
    # Rapper tags
    'it\'s lit', 'straight up', 'you know', 'real talk', 'no cap',
    # Short fillers
    'uh', 'um', 'huh', 'hmm', 'mhm', 'nah', 'na',
    # Laughs
    'ha', 'haha', 'heh',
}

# Words that are almost always adlibs when standalone
DEFINITE_ADLIBS = {'yeah', 'yuh', 'what', 'uh', 'skrrt', 'brr', 'woo', 'sheesh', 'gang'}


@dataclass
class AdlibCandidate:
    """A potential adlib detection"""
    text: str
    start_time: float
    end_time: float
    confidence: float  # 0-1, how likely this is an adlib
    reason: str  # Why we think it's an adlib

    # Audio features
    panning: float  # -1 (left) to 1 (right), 0 = center
    energy_ratio: float  # Relative to surrounding vocals
    duration: float


def detect_adlibs_from_text(
    words: List[dict],  # Whisper word output with start/end times
    main_vocal_gaps: List[Tuple[float, float]] = None
) -> Tuple[List[AdlibCandidate], List[dict]]:
    """
    Detect adlibs based on text content and timing.

    Returns:
        (adlib_candidates, main_vocal_words)
    """
    adlibs = []
    main_words = []

    for i, word in enumerate(words):
        text = word.get('word', '').lower().strip()
        start = word.get('start', 0)
        end = word.get('end', 0)
        duration = end - start

        confidence = 0.0
        reasons = []

        # Check vocabulary
        if text in DEFINITE_ADLIBS:
            confidence += 0.6
            reasons.append("definite_adlib_word")
        elif text in ADLIB_VOCABULARY:
            confidence += 0.4
            reasons.append("adlib_vocabulary")

        # Check duration (adlibs are typically short)
        if duration < 0.3:
            confidence += 0.2
            reasons.append("short_duration")

        # Check if in a gap (requires main_vocal_gaps)
        if main_vocal_gaps:
            for gap_start, gap_end in main_vocal_gaps:
                if gap_start <= start <= gap_end:
                    confidence += 0.3
                    reasons.append("in_vocal_gap")
                    break

        # Check for repetition (same word repeated = likely adlib)
        if i > 0:
            prev_text = words[i-1].get('word', '').lower().strip()
            if text == prev_text and text in ADLIB_VOCABULARY:
                confidence += 0.2
                reasons.append("repeated")

        # Classify
        if confidence >= 0.5:
            adlibs.append(AdlibCandidate(
                text=text,
                start_time=start,
                end_time=end,
                confidence=min(1.0, confidence),
                reason="+".join(reasons),
                panning=0.0,  # Will be filled by audio analysis
                energy_ratio=1.0,
                duration=duration
            ))
        else:
            main_words.append(word)

    return adlibs, main_words


def analyze_stereo_panning(
    audio_path: str,
    time_regions: List[Tuple[float, float]]
) -> List[float]:
    """
    Analyze stereo panning for given time regions.

    Returns panning values: -1 (left) to 1 (right), 0 = center
    """
    y, sr = librosa.load(audio_path, mono=False)

    # If mono, everything is centered
    if y.ndim == 1:
        return [0.0] * len(time_regions)

    left = y[0]
    right = y[1]

    panning_values = []

    for start, end in time_regions:
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        # Get segment
        left_seg = left[start_sample:end_sample]
        right_seg = right[start_sample:end_sample]

        # Calculate energy in each channel
        left_energy = np.sqrt(np.mean(left_seg ** 2)) if len(left_seg) > 0 else 0
        right_energy = np.sqrt(np.mean(right_seg ** 2)) if len(right_seg) > 0 else 0

        # Calculate panning (-1 to 1)
        total = left_energy + right_energy
        if total > 0:
            panning = (right_energy - left_energy) / total
        else:
            panning = 0.0

        panning_values.append(panning)

    return panning_values


def detect_adlibs_with_audio(
    audio_path: str,
    words: List[dict],
    panning_threshold: float = 0.3  # How far from center to consider panned
) -> Tuple[List[AdlibCandidate], List[dict]]:
    """
    Full adlib detection using both text and audio features.
    """
    # First pass: text-based detection
    text_adlibs, main_words = detect_adlibs_from_text(words)

    # Get panning for all words
    all_regions = [(w.get('start', 0), w.get('end', 0)) for w in words]
    panning_values = analyze_stereo_panning(audio_path, all_regions)

    # Second pass: check panning for borderline cases
    final_adlibs = []
    final_main = []

    for i, word in enumerate(words):
        panning = panning_values[i]
        text = word.get('word', '').lower().strip()

        # Check if already identified as adlib
        existing_adlib = None
        for adlib in text_adlibs:
            if adlib.start_time == word.get('start'):
                existing_adlib = adlib
                break

        if existing_adlib:
            # Update with panning info
            existing_adlib.panning = panning
            if abs(panning) > panning_threshold:
                existing_adlib.confidence = min(1.0, existing_adlib.confidence + 0.2)
                existing_adlib.reason += "+panned"
            final_adlibs.append(existing_adlib)

        elif abs(panning) > panning_threshold:
            # Panned but not caught by text detection
            # Could be an adlib we missed
            if text in ADLIB_VOCABULARY or len(text) <= 3:
                final_adlibs.append(AdlibCandidate(
                    text=text,
                    start_time=word.get('start', 0),
                    end_time=word.get('end', 0),
                    confidence=0.5 + abs(panning) * 0.3,
                    reason="panned_vocal",
                    panning=panning,
                    energy_ratio=1.0,
                    duration=word.get('end', 0) - word.get('start', 0)
                ))
            else:
                final_main.append(word)
        else:
            final_main.append(word)

    return final_adlibs, final_main


def calculate_adlib_density(
    adlibs: List[AdlibCandidate],
    total_duration: float,
    num_bars: int
) -> dict:
    """
    Calculate adlib statistics.
    """
    if not adlibs:
        return {
            'total_adlibs': 0,
            'adlibs_per_bar': 0.0,
            'adlib_ratio': 0.0,
            'most_common': [],
            'panning_distribution': {'left': 0, 'center': 0, 'right': 0}
        }

    # Count by text
    from collections import Counter
    text_counts = Counter(a.text for a in adlibs)

    # Panning distribution
    panning_dist = {'left': 0, 'center': 0, 'right': 0}
    for a in adlibs:
        if a.panning < -0.2:
            panning_dist['left'] += 1
        elif a.panning > 0.2:
            panning_dist['right'] += 1
        else:
            panning_dist['center'] += 1

    return {
        'total_adlibs': len(adlibs),
        'adlibs_per_bar': len(adlibs) / num_bars if num_bars > 0 else 0,
        'adlib_ratio': len(adlibs) / (len(adlibs) + 1),  # Rough ratio
        'most_common': text_counts.most_common(5),
        'panning_distribution': panning_dist,
        'avg_confidence': np.mean([a.confidence for a in adlibs])
    }
