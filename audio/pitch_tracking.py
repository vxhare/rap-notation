"""
Pitch Tracking for Melodicity Detection

Detect singing vs rapping by analyzing pitch characteristics:
- Singing: sustained pitches, clear notes, high pitch stability
- Rapping: percussive, pitch varies with speech intonation, less stable

Uses librosa's pyin (probabilistic YIN) algorithm.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class PitchFrame:
    """Pitch data for one time frame"""
    time: float
    f0_hz: Optional[float]      # Fundamental frequency (None if unvoiced)
    confidence: float           # How confident we are in this pitch
    midi_note: Optional[int]    # Quantized to nearest MIDI note
    is_voiced: bool             # True if pitched sound detected


@dataclass
class SyllablePitch:
    """Pitch analysis for one syllable"""
    syllable_id: str
    start_sec: float
    end_sec: float

    # Pitch stats
    mean_f0: Optional[float]
    median_f0: Optional[float]
    std_f0: float               # Standard deviation - high = unstable
    min_f0: Optional[float]
    max_f0: Optional[float]
    pitch_range_semitones: float

    # Melodicity indicators
    voiced_ratio: float         # % of frames with detected pitch
    pitch_stability: float      # 0 = chaotic, 1 = stable sustained note
    melodicity_score: float     # 0 = pure rap, 1 = pure singing

    # Classification
    is_sung: bool
    is_rapped: bool


@dataclass
class MelodicityProfile:
    """Overall melodicity analysis for a track"""
    overall_melodicity: float   # 0-1 average
    melodicity_variance: float  # How much it changes

    sung_ratio: float           # % of syllables classified as sung
    rapped_ratio: float         # % classified as rapped
    mixed_ratio: float          # % in between

    mean_pitch_hz: float
    pitch_range_semitones: float

    # Style classification
    style: str                  # "rap", "melodic_rap", "sing_rap", "singing"
    style_confidence: float


def extract_pitch_contour(
    audio_path: str,
    fmin: float = 65.0,         # C2 - low male voice
    fmax: float = 600.0,        # Above typical singing
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Extract pitch contour from audio using pYIN algorithm.

    Returns:
        times: Time points in seconds
        f0: Fundamental frequencies (Hz), NaN for unvoiced
        voiced_prob: Probability of being voiced (0-1)
        sr: Sample rate
    """
    import librosa

    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)

    # Extract pitch using pYIN
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
        fill_na=np.nan
    )

    # Get time points
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    return times, f0, voiced_prob, sr


def hz_to_midi(f0_hz: float) -> int:
    """Convert frequency to MIDI note number"""
    if f0_hz is None or np.isnan(f0_hz) or f0_hz <= 0:
        return None
    return int(round(12 * np.log2(f0_hz / 440.0) + 69))


def midi_to_note_name(midi: int) -> str:
    """Convert MIDI note to name (e.g., 60 -> C4)"""
    if midi is None:
        return "?"
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi // 12) - 1
    note = notes[midi % 12]
    return f"{note}{octave}"


def analyze_syllable_pitch(
    times: np.ndarray,
    f0: np.ndarray,
    voiced_prob: np.ndarray,
    start_sec: float,
    end_sec: float,
    syllable_id: str = ""
) -> SyllablePitch:
    """
    Analyze pitch characteristics for a single syllable.
    """
    # Find frames within syllable time range
    mask = (times >= start_sec) & (times <= end_sec)
    syl_f0 = f0[mask]
    syl_prob = voiced_prob[mask]

    # Get voiced frames only
    voiced_mask = ~np.isnan(syl_f0)
    voiced_f0 = syl_f0[voiced_mask]

    # Calculate stats
    if len(voiced_f0) > 0:
        mean_f0 = float(np.mean(voiced_f0))
        median_f0 = float(np.median(voiced_f0))
        std_f0 = float(np.std(voiced_f0))
        min_f0 = float(np.min(voiced_f0))
        max_f0 = float(np.max(voiced_f0))

        # Pitch range in semitones
        if min_f0 > 0:
            pitch_range = 12 * np.log2(max_f0 / min_f0)
        else:
            pitch_range = 0
    else:
        mean_f0 = None
        median_f0 = None
        std_f0 = 0
        min_f0 = None
        max_f0 = None
        pitch_range = 0

    # Voiced ratio
    voiced_ratio = len(voiced_f0) / len(syl_f0) if len(syl_f0) > 0 else 0

    # Pitch stability (inverse of coefficient of variation)
    if mean_f0 and mean_f0 > 0 and std_f0 > 0:
        cv = std_f0 / mean_f0  # Coefficient of variation
        pitch_stability = max(0, 1 - cv * 5)  # Scale to 0-1
    else:
        pitch_stability = 0 if voiced_ratio < 0.3 else 0.5

    # Melodicity score combines multiple factors
    melodicity = calculate_melodicity(
        voiced_ratio=voiced_ratio,
        pitch_stability=pitch_stability,
        pitch_range_semitones=pitch_range,
        duration_sec=end_sec - start_sec
    )

    # Classification
    is_sung = melodicity > 0.6
    is_rapped = melodicity < 0.3

    return SyllablePitch(
        syllable_id=syllable_id,
        start_sec=start_sec,
        end_sec=end_sec,
        mean_f0=mean_f0,
        median_f0=median_f0,
        std_f0=std_f0,
        min_f0=min_f0,
        max_f0=max_f0,
        pitch_range_semitones=pitch_range,
        voiced_ratio=voiced_ratio,
        pitch_stability=pitch_stability,
        melodicity_score=melodicity,
        is_sung=is_sung,
        is_rapped=is_rapped
    )


def calculate_melodicity(
    voiced_ratio: float,
    pitch_stability: float,
    pitch_range_semitones: float,
    duration_sec: float
) -> float:
    """
    Calculate melodicity score (0 = rap, 1 = singing).

    Singing characteristics:
    - High voiced ratio (sustained pitch)
    - High pitch stability (holding notes)
    - Moderate pitch range (melodic movement)
    - Longer durations (held notes)

    Rap characteristics:
    - Lower voiced ratio (percussive)
    - Lower stability (pitch follows speech)
    - Narrow range OR very wide (speech intonation)
    - Shorter, punchier syllables
    """
    # Factor 1: Voiced ratio (singing is more pitched)
    # 0.8+ voiced = very sung, <0.4 = more percussive
    voiced_score = min(1.0, voiced_ratio / 0.8)

    # Factor 2: Pitch stability
    stability_score = pitch_stability

    # Factor 3: Duration (longer = more sung)
    # Normalize: 200ms is typical rap syllable, 400ms+ is sung
    duration_score = min(1.0, (duration_sec * 1000 - 100) / 300)
    duration_score = max(0, duration_score)

    # Factor 4: Pitch range (moderate range = melodic)
    # <1 semitone = monotone (could be either)
    # 2-5 semitones = melodic
    # >8 semitones = speech inflection or very melodic
    if pitch_range_semitones < 1:
        range_score = 0.3  # Ambiguous
    elif pitch_range_semitones < 5:
        range_score = 0.7  # Melodic
    elif pitch_range_semitones < 8:
        range_score = 0.5  # Could be either
    else:
        range_score = 0.3  # Wide range often speech-like

    # Weighted combination
    melodicity = (
        voiced_score * 0.35 +
        stability_score * 0.30 +
        duration_score * 0.20 +
        range_score * 0.15
    )

    return min(1.0, max(0.0, melodicity))


def analyze_track_melodicity(
    audio_path: str,
    syllables: List[dict] = None,
    verbose: bool = False
) -> Tuple[MelodicityProfile, List[SyllablePitch]]:
    """
    Full melodicity analysis for a track.

    Args:
        audio_path: Path to audio file
        syllables: Optional list of syllable dicts with 'start' and 'end' times
        verbose: Print progress

    Returns:
        MelodicityProfile and list of per-syllable pitch data
    """
    if verbose:
        print("Extracting pitch contour...")

    times, f0, voiced_prob, sr = extract_pitch_contour(audio_path)

    if verbose:
        voiced_frames = np.sum(~np.isnan(f0))
        print(f"  Frames: {len(f0)}, Voiced: {voiced_frames} ({voiced_frames/len(f0)*100:.1f}%)")

    # Analyze per-syllable if syllables provided
    syllable_pitches = []

    if syllables:
        if verbose:
            print(f"Analyzing {len(syllables)} syllables...")

        for syl in syllables:
            syl_pitch = analyze_syllable_pitch(
                times, f0, voiced_prob,
                start_sec=syl.get('start', 0),
                end_sec=syl.get('end', 0),
                syllable_id=syl.get('id', '')
            )
            syllable_pitches.append(syl_pitch)
    else:
        # Analyze in fixed windows (100ms)
        window_size = 0.1
        for start in np.arange(0, times[-1], window_size):
            syl_pitch = analyze_syllable_pitch(
                times, f0, voiced_prob,
                start_sec=start,
                end_sec=start + window_size,
                syllable_id=f"window_{int(start*10)}"
            )
            syllable_pitches.append(syl_pitch)

    # Aggregate stats
    melodicities = [sp.melodicity_score for sp in syllable_pitches]
    overall_melodicity = np.mean(melodicities) if melodicities else 0
    melodicity_variance = np.var(melodicities) if melodicities else 0

    sung_count = sum(1 for sp in syllable_pitches if sp.is_sung)
    rapped_count = sum(1 for sp in syllable_pitches if sp.is_rapped)
    mixed_count = len(syllable_pitches) - sung_count - rapped_count

    total = len(syllable_pitches) or 1
    sung_ratio = sung_count / total
    rapped_ratio = rapped_count / total
    mixed_ratio = mixed_count / total

    # Overall pitch stats
    voiced_f0 = f0[~np.isnan(f0)]
    if len(voiced_f0) > 0:
        mean_pitch = float(np.mean(voiced_f0))
        min_pitch = float(np.min(voiced_f0))
        max_pitch = float(np.max(voiced_f0))
        pitch_range = 12 * np.log2(max_pitch / min_pitch) if min_pitch > 0 else 0
    else:
        mean_pitch = 0
        pitch_range = 0

    # Style classification
    if overall_melodicity > 0.65:
        style = "singing"
        confidence = min(1.0, (overall_melodicity - 0.5) * 3)
    elif overall_melodicity > 0.45:
        style = "sing_rap"
        confidence = 1.0 - abs(overall_melodicity - 0.55) * 3
    elif overall_melodicity > 0.25:
        style = "melodic_rap"
        confidence = 1.0 - abs(overall_melodicity - 0.35) * 3
    else:
        style = "rap"
        confidence = min(1.0, (0.4 - overall_melodicity) * 3)

    profile = MelodicityProfile(
        overall_melodicity=overall_melodicity,
        melodicity_variance=melodicity_variance,
        sung_ratio=sung_ratio,
        rapped_ratio=rapped_ratio,
        mixed_ratio=mixed_ratio,
        mean_pitch_hz=mean_pitch,
        pitch_range_semitones=pitch_range,
        style=style,
        style_confidence=confidence
    )

    if verbose:
        print(f"\nMelodicity Profile:")
        print(f"  Overall: {overall_melodicity:.2f}")
        print(f"  Style: {style} (confidence: {confidence:.2f})")
        print(f"  Sung: {sung_ratio*100:.0f}%, Rapped: {rapped_ratio*100:.0f}%, Mixed: {mixed_ratio*100:.0f}%")
        print(f"  Mean pitch: {mean_pitch:.0f}Hz ({midi_to_note_name(hz_to_midi(mean_pitch))})")
        print(f"  Pitch range: {pitch_range:.1f} semitones")

    return profile, syllable_pitches


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pitch_tracking.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    print("=" * 60)
    print("PITCH TRACKING / MELODICITY ANALYSIS")
    print("=" * 60)
    print(f"File: {audio_file}")
    print()

    profile, syllable_pitches = analyze_track_melodicity(
        audio_file,
        syllables=None,  # No syllables, analyze in windows
        verbose=True
    )

    print("\n" + "=" * 60)
    print("MELODICITY BY SECTION")
    print("=" * 60)

    # Group into ~10 second chunks
    chunk_size = 100  # 100 windows = 10 seconds
    for i in range(0, len(syllable_pitches), chunk_size):
        chunk = syllable_pitches[i:i+chunk_size]
        if not chunk:
            continue

        avg_mel = np.mean([sp.melodicity_score for sp in chunk])
        start_time = chunk[0].start_sec
        end_time = chunk[-1].end_sec

        bar = "#" * int(avg_mel * 20)
        label = "SING" if avg_mel > 0.6 else ("melodic" if avg_mel > 0.35 else "rap")

        print(f"{start_time:5.1f}s - {end_time:5.1f}s: {avg_mel:.2f} {bar:<20} [{label}]")
