"""
Vocal Double Detection v2 - Accounting for Beat Vocals

The problem: Demucs "vocals" includes both:
- Main artist vocals (and their doubles)
- Beat/production vocals (samples, chops, background)

To detect ACTUAL doubles, we need to distinguish:
1. Same voice doubled (what we want) vs Different voices layered
2. Lead vocal vs beat vocal

Approaches:
1. PITCH CONSISTENCY - A true double has same pitch as main vocal
2. LYRIC ALIGNMENT - Beat vocals won't match transcribed lyrics
3. SPECTRAL SIMILARITY - True doubles are spectrally similar to main
4. TIMING PATTERN - Beat vocals repeat rhythmically, doubles follow lyrics
5. SPEAKER EMBEDDING - Same speaker vs different speaker
"""

import numpy as np
import librosa
from scipy import signal
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


@dataclass
class VocalDoubleV2:
    """Improved vocal double detection with beat vocal filtering"""
    start_time: float
    end_time: float
    word_or_phrase: str
    confidence: float

    # New fields for filtering beat vocals
    pitch_match_score: float      # How well pitch matches main vocal (0-1)
    spectral_similarity: float    # Spectral similarity to main vocal (0-1)
    lyric_aligned: bool           # Does this align with transcribed lyrics?
    likely_beat_vocal: bool       # Is this probably a beat vocal, not a double?

    # Original fields
    stereo_width: float
    amplitude_boost_db: float


def extract_pitch_contour(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract pitch contour using pyin."""
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'),
        sr=sr
    )
    times = librosa.times_like(f0, sr=sr)
    return times, f0


def compute_spectral_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute spectral features for similarity comparison."""
    # MFCCs are good for voice similarity
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)  # Average over time


def get_main_vocal_baseline(
    y: np.ndarray,
    sr: int,
    syllables: List[dict]
) -> dict:
    """
    Establish baseline characteristics of the main vocal
    using sections where we KNOW it's the main artist (from lyrics alignment).
    """
    # Get sections where lyrics are aligned (high confidence main vocal)
    baseline_segments = []

    for syl in syllables:
        start = int(syl.get('start', 0) * sr)
        end = int(syl.get('end', 0) * sr)
        if end > start and end < len(y):
            baseline_segments.append(y[start:end])

    if not baseline_segments:
        return None

    # Concatenate baseline segments
    baseline_audio = np.concatenate(baseline_segments)

    # Extract features
    _, pitch = extract_pitch_contour(baseline_audio, sr)
    spectral = compute_spectral_features(baseline_audio, sr)

    # Get pitch statistics (ignoring NaN/unvoiced)
    valid_pitch = pitch[~np.isnan(pitch)]

    return {
        'mean_pitch': np.mean(valid_pitch) if len(valid_pitch) > 0 else 0,
        'std_pitch': np.std(valid_pitch) if len(valid_pitch) > 0 else 0,
        'pitch_range': (np.percentile(valid_pitch, 10), np.percentile(valid_pitch, 90)) if len(valid_pitch) > 0 else (0, 0),
        'spectral_features': spectral,
        'mean_amplitude': np.sqrt(np.mean(baseline_audio ** 2))
    }


def is_likely_beat_vocal(
    segment: np.ndarray,
    sr: int,
    main_baseline: dict,
    syllables: List[dict],
    segment_start: float,
    segment_end: float
) -> Tuple[bool, float, float, bool]:
    """
    Determine if a segment is likely a beat vocal (not a true double).

    Returns: (is_beat_vocal, pitch_match, spectral_sim, lyric_aligned)
    """
    if main_baseline is None:
        return False, 0.5, 0.5, True

    # 1. Check pitch match
    _, seg_pitch = extract_pitch_contour(segment, sr)
    valid_pitch = seg_pitch[~np.isnan(seg_pitch)]

    if len(valid_pitch) > 0:
        seg_mean_pitch = np.mean(valid_pitch)
        main_low, main_high = main_baseline['pitch_range']

        # Is pitch in the main vocal's range?
        if main_low > 0 and main_high > 0:
            if main_low <= seg_mean_pitch <= main_high:
                pitch_match = 1.0
            else:
                # How far outside the range?
                if seg_mean_pitch < main_low:
                    distance = (main_low - seg_mean_pitch) / main_low
                else:
                    distance = (seg_mean_pitch - main_high) / main_high
                pitch_match = max(0, 1 - distance)
        else:
            pitch_match = 0.5
    else:
        pitch_match = 0.5

    # 2. Check spectral similarity
    seg_spectral = compute_spectral_features(segment, sr)
    main_spectral = main_baseline['spectral_features']

    # Cosine similarity
    spectral_sim = 1 - cosine(seg_spectral, main_spectral)
    spectral_sim = max(0, min(1, spectral_sim))

    # 3. Check lyric alignment
    # Does this segment overlap with any transcribed syllables?
    lyric_aligned = False
    for syl in syllables:
        syl_start = syl.get('start', 0)
        syl_end = syl.get('end', 0)

        # Check overlap
        if syl_start < segment_end and syl_end > segment_start:
            lyric_aligned = True
            break

    # 4. Decision: is it likely a beat vocal?
    # Beat vocals typically:
    # - Have different pitch than main vocal
    # - Have different spectral characteristics
    # - Don't align with lyrics

    beat_vocal_score = 0

    if pitch_match < 0.5:
        beat_vocal_score += 0.4  # Different pitch range

    if spectral_sim < 0.7:
        beat_vocal_score += 0.3  # Different voice characteristics

    if not lyric_aligned:
        beat_vocal_score += 0.3  # Not aligned with lyrics

    is_beat_vocal = beat_vocal_score >= 0.5

    return is_beat_vocal, pitch_match, spectral_sim, lyric_aligned


def detect_doubles_v2(
    audio_path: str,
    syllables: List[dict],
    bpm: float,
    sensitivity: str = "normal",
    verbose: bool = True
) -> List[VocalDoubleV2]:
    """
    Detect vocal doubles while filtering out beat vocals.

    This version:
    1. Establishes main vocal baseline from lyric-aligned sections
    2. For each candidate double, checks if it matches main vocal characteristics
    3. Filters out sections that are likely beat vocals
    """
    if verbose:
        print("  Loading audio...")

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = len(y) / sr

    if verbose:
        print("  Establishing main vocal baseline from lyrics...")

    main_baseline = get_main_vocal_baseline(y, sr, syllables)

    if main_baseline and verbose:
        print(f"    Main vocal pitch range: {main_baseline['pitch_range'][0]:.0f}-{main_baseline['pitch_range'][1]:.0f} Hz")

    # Load stereo for width analysis
    y_stereo, _ = librosa.load(audio_path, sr=22050, mono=False)
    if y_stereo.ndim == 1:
        y_stereo = np.array([y_stereo, y_stereo])

    if verbose:
        print("  Computing features...")

    # Compute amplitude and stereo width (same as before)
    hop_length = 512
    frame_length = 2048

    # RMS amplitude
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Stereo width
    n_frames = len(rms)
    width = np.zeros(n_frames)

    left = y_stereo[0]
    right = y_stereo[1]

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        if end > len(left):
            break

        l_frame = left[start:end]
        r_frame = right[start:end]

        if np.std(l_frame) > 0 and np.std(r_frame) > 0:
            corr = np.corrcoef(l_frame, r_frame)[0, 1]
            width[i] = 1.0 - max(0, corr)

    # Baseline estimation
    loud_mask = rms > np.percentile(rms, 10)
    baseline_rms = np.mean(rms[loud_mask])
    baseline_width = np.mean(width[loud_mask])

    # Convert to dB
    rms_db = 20 * np.log10(rms / (baseline_rms + 1e-10))
    width_deviation = width - baseline_width

    # Sensitivity thresholds
    thresholds = {
        "low": {"amp_db": 6.0, "width": 0.35},
        "normal": {"amp_db": 4.5, "width": 0.25},
        "high": {"amp_db": 3.0, "width": 0.15},
    }
    thresh = thresholds.get(sensitivity, thresholds["normal"])

    # Find candidate doubles
    is_candidate = (rms_db > thresh["amp_db"]) | (width_deviation > thresh["width"])

    # Smooth
    from scipy.ndimage import uniform_filter1d
    is_candidate = uniform_filter1d(is_candidate.astype(float), size=5) > 0.5

    # Find contiguous regions
    candidates = []
    in_region = False
    start_idx = 0

    for i, is_c in enumerate(is_candidate):
        if is_c and not in_region:
            in_region = True
            start_idx = i
        elif not is_c and in_region:
            in_region = False
            if times[i] - times[start_idx] >= 0.1:
                candidates.append((times[start_idx], times[i], start_idx, i))

    if verbose:
        print(f"  Found {len(candidates)} candidate regions")
        print("  Filtering beat vocals...")

    # Filter candidates
    doubles = []
    beat_vocal_count = 0

    for start_time, end_time, start_idx, end_idx in candidates:
        # Extract segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]

        if len(segment) < 512:
            continue

        # Check if likely beat vocal
        is_beat, pitch_match, spectral_sim, lyric_aligned = is_likely_beat_vocal(
            segment, sr, main_baseline, syllables, start_time, end_time
        )

        if is_beat:
            beat_vocal_count += 1
            continue  # Skip beat vocals

        # Get metrics
        region_amp_db = np.mean(rms_db[start_idx:end_idx])
        region_width = np.mean(width[start_idx:end_idx])

        # Find word
        word = ""
        for syl in syllables:
            if syl.get('start', 0) >= start_time - 0.1 and syl.get('end', 0) <= end_time + 0.1:
                word += syl.get('text', '') + " "

        # Confidence based on how well it matches main vocal
        confidence = (pitch_match * 0.4 + spectral_sim * 0.4 + (0.2 if lyric_aligned else 0))

        doubles.append(VocalDoubleV2(
            start_time=start_time,
            end_time=end_time,
            word_or_phrase=word.strip(),
            confidence=confidence,
            pitch_match_score=pitch_match,
            spectral_similarity=spectral_sim,
            lyric_aligned=lyric_aligned,
            likely_beat_vocal=False,
            stereo_width=region_width,
            amplitude_boost_db=region_amp_db
        ))

    if verbose:
        print(f"  Filtered out {beat_vocal_count} likely beat vocals")
        print(f"  Remaining true doubles: {len(doubles)}")

    return doubles


def compare_detection_methods(
    audio_path: str,
    vocals_path: str,
    syllables: List[dict],
    bpm: float
):
    """
    Compare v1 (naive) vs v2 (beat-vocal-aware) detection.
    """
    from audio.vocal_double_detection import detect_vocal_doubles

    print("=" * 60)
    print("COMPARISON: Naive vs Beat-Vocal-Aware Detection")
    print("=" * 60)

    # V1: Naive detection on isolated vocals
    print("\n[V1] Naive detection on isolated vocals...")
    v1_result = detect_vocal_doubles(
        vocals_path,
        bpm=bpm,
        sensitivity='normal',
        verbose=False
    )

    # V2: Beat-vocal-aware detection
    print("\n[V2] Beat-vocal-aware detection...")
    v2_doubles = detect_doubles_v2(
        vocals_path,
        syllables=syllables,
        bpm=bpm,
        sensitivity='normal',
        verbose=True
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"""
V1 (Naive):
  Doubles found: {len(v1_result.doubles)}
  Doubled time: {v1_result.doubled_ratio*100:.1f}%

V2 (Beat-vocal-aware):
  Doubles found: {len(v2_doubles)}
  (Filtered out beat vocals based on pitch/spectral mismatch)
""")

    if v2_doubles:
        print("Top V2 doubles (likely TRUE doubles):")
        sorted_doubles = sorted(v2_doubles, key=lambda d: d.confidence, reverse=True)[:10]
        for i, d in enumerate(sorted_doubles, 1):
            print(f"  {i}. {d.start_time:.2f}s - conf={d.confidence:.2f}, "
                  f"pitch_match={d.pitch_match_score:.2f}, "
                  f"spectral_sim={d.spectral_similarity:.2f}, "
                  f"lyric={d.lyric_aligned}")

    return v1_result, v2_doubles
