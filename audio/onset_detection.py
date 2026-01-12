"""
Rhythmic Syllable Detection

Detect actual vocal events (rhythmic syllables) from audio,
NOT linguistic syllables from dictionary rules.

A "rhythmic syllable" is when the rapper's voice hits a new sound -
what actually lands on the beat grid.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class RhythmicEvent:
    """A detected vocal event (rhythmic syllable)"""
    index: int
    start_sec: float
    end_sec: float
    duration_ms: float
    strength: float  # Onset strength (0-1)
    text: Optional[str] = None  # Filled in later by alignment


def detect_rhythmic_syllables(
    audio_path: str,
    min_silence_ms: float = 50,
    onset_threshold: float = 0.1,
    backtrack: bool = True
) -> list:
    """
    Detect rhythmic syllables (vocal onsets) from audio.

    This finds WHERE vocal events happen, not what they are.
    Text is aligned separately.

    Args:
        audio_path: Path to audio file
        min_silence_ms: Minimum gap to consider a new syllable
        onset_threshold: Sensitivity (lower = more onsets)
        backtrack: Adjust onsets to local energy minimum

    Returns:
        List of RhythmicEvent objects
    """
    import librosa

    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)

    # Compute onset envelope
    # Use multiple features for robustness
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        aggregate=np.median,  # More robust than mean
        fmax=8000,  # Focus on vocal range
        n_mels=128
    )

    # Detect onsets
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        units='time',
        backtrack=backtrack,
        delta=onset_threshold
    )

    # Filter out onsets that are too close together
    min_gap = min_silence_ms / 1000
    filtered_onsets = [onsets[0]] if len(onsets) > 0 else []

    for onset in onsets[1:]:
        if onset - filtered_onsets[-1] >= min_gap:
            filtered_onsets.append(onset)

    # Build events with durations
    events = []
    for i, start in enumerate(filtered_onsets):
        # End is start of next onset, or end of audio
        if i < len(filtered_onsets) - 1:
            end = filtered_onsets[i + 1]
        else:
            end = len(y) / sr

        # Get onset strength at this point
        frame = librosa.time_to_frames(start, sr=sr)
        strength = onset_env[frame] / onset_env.max() if frame < len(onset_env) else 0.5

        events.append(RhythmicEvent(
            index=i,
            start_sec=start,
            end_sec=end,
            duration_ms=(end - start) * 1000,
            strength=float(strength)
        ))

    return events


def detect_vocal_onsets_aggressive(
    audio_path: str,
    hop_length: int = 512,
    n_mels: int = 128
) -> list:
    """
    More aggressive onset detection tuned for rap vocals.
    Uses spectral flux in vocal frequency range.
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=22050)

    # Compute mel spectrogram focused on vocal range (100Hz - 4kHz)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        fmin=100,
        fmax=4000,
        hop_length=hop_length
    )

    # Convert to dB
    S_db = librosa.power_to_db(S, ref=np.max)

    # Compute spectral flux (change between frames)
    flux = np.diff(S_db, axis=1)
    flux = np.maximum(0, flux)  # Only positive changes (onsets, not offsets)
    flux = np.sum(flux, axis=0)  # Sum across frequency bands

    # Normalize
    flux = flux / flux.max() if flux.max() > 0 else flux

    # Peak picking
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(
        flux,
        height=0.1,
        distance=int(0.05 * sr / hop_length),  # Min 50ms apart
        prominence=0.05
    )

    # Convert to times
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    # Build events
    events = []
    for i, start in enumerate(times):
        end = times[i + 1] if i < len(times) - 1 else len(y) / sr

        events.append(RhythmicEvent(
            index=i,
            start_sec=float(start),
            end_sec=float(end),
            duration_ms=float((end - start) * 1000),
            strength=float(properties['peak_heights'][i]) if 'peak_heights' in properties else 0.5
        ))

    return events


def align_text_to_events(
    events: list,
    whisper_words: list,
    tolerance_ms: float = 100
) -> list:
    """
    Align Whisper's word timestamps to detected rhythmic events.

    This distributes the transcribed text across the actual
    vocal onsets we detected.

    Args:
        events: List of RhythmicEvent from onset detection
        whisper_words: List of {"text": str, "start": float, "end": float}
        tolerance_ms: How close a word must be to an event to match

    Returns:
        Events with text filled in
    """
    tolerance = tolerance_ms / 1000

    # Create a copy of events to modify
    aligned_events = [RhythmicEvent(
        index=e.index,
        start_sec=e.start_sec,
        end_sec=e.end_sec,
        duration_ms=e.duration_ms,
        strength=e.strength,
        text=None
    ) for e in events]

    # For each Whisper word, find the closest event
    word_idx = 0

    for event in aligned_events:
        # Find words that fall within this event's time window
        matching_words = []

        for word in whisper_words:
            word_start = word['start']
            word_end = word['end']

            # Check if word overlaps with event
            if (word_start <= event.end_sec + tolerance and
                word_end >= event.start_sec - tolerance):
                matching_words.append(word['text'])

        if matching_words:
            event.text = ' '.join(matching_words).strip()

    # Filter out events with no text (likely instrumental/silence)
    voiced_events = [e for e in aligned_events if e.text]

    # Re-index
    for i, e in enumerate(voiced_events):
        e.index = i

    return voiced_events


def split_words_to_rhythmic_syllables(
    whisper_words: list,
    events: list
) -> list:
    """
    Smarter alignment: split multi-syllable words across multiple events.

    If Whisper gives us "beautiful" at t=1.0, and we detect 3 onsets
    at t=1.0, 1.1, 1.2 - split "beautiful" into "beau", "ti", "ful".
    """
    import pyphen

    # Initialize hyphenator for syllable splitting
    dic = pyphen.Pyphen(lang='en_US')

    result_events = []
    word_idx = 0

    for event in events:
        event_start = event.start_sec
        event_end = event.end_sec

        # Find words that overlap with this event
        overlapping_text = []

        while word_idx < len(whisper_words):
            word = whisper_words[word_idx]
            word_start = word['start']
            word_end = word['end']

            # Word is before event - skip
            if word_end < event_start - 0.05:
                word_idx += 1
                continue

            # Word is after event - done with this event
            if word_start > event_end + 0.05:
                break

            # Word overlaps with event
            overlapping_text.append(word['text'].strip())
            word_idx += 1

        if overlapping_text:
            text = ' '.join(overlapping_text)
            result_events.append(RhythmicEvent(
                index=len(result_events),
                start_sec=event.start_sec,
                end_sec=event.end_sec,
                duration_ms=event.duration_ms,
                strength=event.strength,
                text=text
            ))

    return result_events


def count_rhythmic_syllables(audio_path: str, verbose: bool = False) -> dict:
    """
    Quick analysis: how many rhythmic syllables in this track?

    Returns stats about detected vocal events.
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)

    # Detect with both methods
    events_standard = detect_rhythmic_syllables(audio_path)
    events_aggressive = detect_vocal_onsets_aggressive(audio_path)

    if verbose:
        print(f"Duration: {duration:.1f}s")
        print(f"Standard detection: {len(events_standard)} events")
        print(f"Aggressive detection: {len(events_aggressive)} events")
        print(f"Events/second: {len(events_standard)/duration:.1f} (std), {len(events_aggressive)/duration:.1f} (agg)")

    return {
        'duration': duration,
        'standard_count': len(events_standard),
        'aggressive_count': len(events_aggressive),
        'events_per_second_std': len(events_standard) / duration,
        'events_per_second_agg': len(events_aggressive) / duration,
        'events_standard': events_standard,
        'events_aggressive': events_aggressive
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python onset_detection.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    print("=" * 60)
    print("RHYTHMIC SYLLABLE DETECTION")
    print("=" * 60)

    stats = count_rhythmic_syllables(audio_file, verbose=True)

    print()
    print("First 20 events (standard):")
    for e in stats['events_standard'][:20]:
        print(f"  {e.start_sec:6.2f}s - {e.end_sec:6.2f}s ({e.duration_ms:5.0f}ms) strength={e.strength:.2f}")
