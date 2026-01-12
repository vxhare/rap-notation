"""
Vocal Double Detection Module

Detects when vocals are doubled/stacked in a track:
- Same lyrics layered on top of itself
- Often louder, wider stereo, different effects
- Usually on emphasis words: line endings, punchlines, rhyme words

Detection approach:
1. Analyze amplitude envelope (doubles are louder)
2. Measure stereo width (doubles often panned L/R)
3. Spectral density (stacked voices = more harmonics)
4. Phase analysis (slight timing differences between takes)
5. Formant analysis (multiple voices = smeared formants)
"""

import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class DoubleType(Enum):
    """Types of vocal doubles"""
    FULL_DOUBLE = "full_double"      # Complete duplicate take
    STACK = "stack"                   # Multiple layered takes
    WHISPER = "whisper"               # Whispered double underneath
    OCTAVE = "octave"                 # Octave up/down double
    DELAY_EFFECT = "delay_effect"     # Slapback/delay creating double
    HARMONY = "harmony"               # Harmonized double
    UNKNOWN = "unknown"


@dataclass
class VocalDouble:
    """A detected vocal double/stack"""
    start_time: float           # When the double starts
    end_time: float             # When it ends
    word_or_phrase: str         # What's being doubled (if aligned)
    confidence: float           # How sure we are (0-1)
    double_type: DoubleType     # Type of double
    stereo_width: float         # How wide (0=mono, 1=full L/R)
    amplitude_boost_db: float   # dB louder than surrounding
    spectral_density: float     # Harmonic density increase
    position_in_bar: float      # Beat position (0-4)
    is_rhyme_word: bool         # Does this participate in rhyme
    is_line_ending: bool        # Is this at phrase end

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class VocalBaseline:
    """Baseline characteristics of single-voice sections"""
    mean_amplitude: float
    std_amplitude: float
    mean_stereo_width: float
    std_stereo_width: float
    mean_spectral_centroid: float
    mean_spectral_bandwidth: float
    mean_spectral_flatness: float


@dataclass
class DoubleDetectionResult:
    """Complete result of double detection"""
    doubles: List[VocalDouble]
    baseline: VocalBaseline
    total_doubled_time: float
    doubled_ratio: float  # What % of track has doubles
    double_density_per_bar: float
    most_common_type: DoubleType


# =============================================================================
# STEM SEPARATION
# =============================================================================

def separate_vocals_demucs(
    audio_path: str,
    output_dir: Optional[str] = None
) -> str:
    """
    Separate vocals from mix using Demucs Python API.

    Returns path to isolated vocal file.
    """
    import tempfile
    from pathlib import Path

    try:
        import torch
        import torchaudio
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except ImportError:
        raise RuntimeError("Demucs not installed. Run: pip install demucs")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="demucs_")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load model
    model = get_model('htdemucs')
    model.eval()

    # Load audio
    wav, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
        sr = model.samplerate

    # Add batch dimension
    wav = wav.unsqueeze(0)

    # Apply model
    with torch.no_grad():
        sources = apply_model(model, wav, progress=True)

    # sources shape: (batch, num_sources, channels, samples)
    # htdemucs sources: drums, bass, other, vocals
    vocals_idx = model.sources.index('vocals')
    vocals = sources[0, vocals_idx]

    # Save vocals
    vocals_path = output_dir / "vocals.wav"

    # Use soundfile for saving (more reliable)
    import soundfile as sf
    sf.write(str(vocals_path), vocals.cpu().numpy().T, sr)

    return str(vocals_path)


def load_audio_stereo(audio_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load audio keeping stereo channels.
    Returns (audio, sr) where audio shape is (2, samples) or (samples,) if mono.
    """
    y, sr_loaded = librosa.load(audio_path, sr=sr, mono=False)
    return y, sr_loaded


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def compute_amplitude_envelope(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute amplitude envelope (RMS) over time.

    Returns (times, rms_values)
    """
    # If stereo, sum channels
    if y.ndim == 2:
        y_mono = np.mean(y, axis=0)
    else:
        y_mono = y

    rms = librosa.feature.rms(y=y_mono, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    return times, rms


def compute_stereo_width(
    y_stereo: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute stereo width over time.

    Width = 1 - correlation(L, R)
    - 0 = mono (L and R identical)
    - 1 = full width (L and R uncorrelated or opposite)

    Returns (times, width_values)
    """
    if y_stereo.ndim != 2 or y_stereo.shape[0] != 2:
        # Mono audio - return zeros
        n_frames = len(y_stereo) // hop_length
        times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
        return times, np.zeros(n_frames)

    left = y_stereo[0]
    right = y_stereo[1]

    n_frames = (len(left) - frame_length) // hop_length + 1
    width = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length

        l_frame = left[start:end]
        r_frame = right[start:end]

        # Correlation coefficient
        if np.std(l_frame) > 0 and np.std(r_frame) > 0:
            corr = np.corrcoef(l_frame, r_frame)[0, 1]
            # Width = 1 - correlation (mono = 0, full stereo = 1)
            width[i] = 1.0 - max(0, corr)  # Clamp negative correlations
        else:
            width[i] = 0.0

    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
    return times, width


def compute_spectral_density(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectral features that indicate vocal stacking.

    Returns (times, spectral_centroid, spectral_bandwidth, spectral_flatness)
    """
    if y.ndim == 2:
        y_mono = np.mean(y, axis=0)
    else:
        y_mono = y

    # Spectral centroid - "brightness"
    centroid = librosa.feature.spectral_centroid(
        y=y_mono, sr=sr, n_fft=frame_length, hop_length=hop_length
    )[0]

    # Spectral bandwidth - width of spectrum
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y_mono, sr=sr, n_fft=frame_length, hop_length=hop_length
    )[0]

    # Spectral flatness - tonality vs noise
    # Stacked vocals have more tonal content (lower flatness)
    flatness = librosa.feature.spectral_flatness(
        y=y_mono, n_fft=frame_length, hop_length=hop_length
    )[0]

    times = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=hop_length)

    return times, centroid, bandwidth, flatness


def compute_phase_coherence(
    y_stereo: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute phase coherence between L/R channels.

    Low coherence can indicate doubled vocals with slight timing differences.

    Returns (times, coherence_values)
    """
    if y_stereo.ndim != 2 or y_stereo.shape[0] != 2:
        n_frames = len(y_stereo) // hop_length
        times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
        return times, np.ones(n_frames)  # Mono = perfect coherence

    left = y_stereo[0]
    right = y_stereo[1]

    # STFT of both channels
    S_left = librosa.stft(left, n_fft=frame_length, hop_length=hop_length)
    S_right = librosa.stft(right, n_fft=frame_length, hop_length=hop_length)

    # Phase difference
    phase_left = np.angle(S_left)
    phase_right = np.angle(S_right)
    phase_diff = phase_left - phase_right

    # Coherence = mean of cos(phase_diff) across frequencies
    # 1 = in phase, 0 = random phase, -1 = out of phase
    coherence = np.mean(np.cos(phase_diff), axis=0)

    times = librosa.frames_to_time(np.arange(len(coherence)), sr=sr, hop_length=hop_length)

    return times, coherence


# =============================================================================
# BASELINE ESTIMATION
# =============================================================================

def estimate_baseline(
    times: np.ndarray,
    amplitude: np.ndarray,
    stereo_width: np.ndarray,
    spectral_centroid: np.ndarray,
    spectral_bandwidth: np.ndarray,
    spectral_flatness: np.ndarray,
    percentile_range: Tuple[int, int] = (25, 75)
) -> VocalBaseline:
    """
    Estimate baseline (single-voice) characteristics.

    Uses the middle 50% of values to avoid outliers (doubles and silence).
    """
    # Filter out very quiet sections (likely silence)
    loud_mask = amplitude > np.percentile(amplitude, 10)

    # Use middle percentile range for baseline
    amp_low, amp_high = np.percentile(amplitude[loud_mask], percentile_range)
    baseline_mask = (amplitude >= amp_low) & (amplitude <= amp_high) & loud_mask

    if np.sum(baseline_mask) < 10:
        # Fallback to all loud sections
        baseline_mask = loud_mask

    return VocalBaseline(
        mean_amplitude=np.mean(amplitude[baseline_mask]),
        std_amplitude=np.std(amplitude[baseline_mask]),
        mean_stereo_width=np.mean(stereo_width[baseline_mask]) if len(stereo_width) == len(baseline_mask) else 0,
        std_stereo_width=np.std(stereo_width[baseline_mask]) if len(stereo_width) == len(baseline_mask) else 0,
        mean_spectral_centroid=np.mean(spectral_centroid[baseline_mask]),
        mean_spectral_bandwidth=np.mean(spectral_bandwidth[baseline_mask]),
        mean_spectral_flatness=np.mean(spectral_flatness[baseline_mask])
    )


# =============================================================================
# DOUBLE DETECTION
# =============================================================================

def detect_doubles_simple(
    times: np.ndarray,
    amplitude: np.ndarray,
    stereo_width: np.ndarray,
    baseline: VocalBaseline,
    amplitude_threshold_db: float = 4.5,  # Raised: need significant boost
    width_threshold: float = 0.25,         # Raised: need clear stereo spread
    min_duration: float = 0.15,            # Raised: ignore very short blips
    require_both: bool = False             # If True, need BOTH amp AND width
) -> List[Tuple[float, float, float, float]]:
    """
    Simple double detection using amplitude + stereo width.

    Returns list of (start_time, end_time, amplitude_boost_db, stereo_width)
    """
    # Convert amplitude to dB relative to baseline
    amplitude_db = 20 * np.log10(amplitude / (baseline.mean_amplitude + 1e-10))

    # Compute width deviation from baseline
    width_deviation = stereo_width - baseline.mean_stereo_width

    # Combined score: high amplitude OR high width suggests double
    # Normalize both to 0-1 range
    amp_score = np.clip(amplitude_db / 10.0, 0, 1)  # 10dB = score of 1
    width_score = np.clip(width_deviation / 0.5, 0, 1)  # 0.5 width increase = score of 1

    # Combined score
    combined_score = np.maximum(amp_score, width_score * 0.8)

    # Threshold for double detection
    amp_above = amplitude_db > amplitude_threshold_db
    width_above = width_deviation > width_threshold

    if require_both:
        # Conservative: need BOTH amplitude AND width increase
        is_double = amp_above & width_above
    else:
        # Standard: either can trigger, but weight amplitude more
        # Only trigger on width alone if it's really high
        is_double = amp_above | (width_deviation > width_threshold * 1.5)

    # Smooth to avoid rapid on/off
    is_double_smooth = uniform_filter1d(is_double.astype(float), size=5) > 0.5

    # Find contiguous regions
    doubles = []
    in_double = False
    start_idx = 0

    for i, is_d in enumerate(is_double_smooth):
        if is_d and not in_double:
            # Start of double
            in_double = True
            start_idx = i
        elif not is_d and in_double:
            # End of double
            in_double = False
            start_time = times[start_idx]
            end_time = times[i]

            if end_time - start_time >= min_duration:
                # Calculate metrics for this region
                region_amp_db = np.mean(amplitude_db[start_idx:i])
                region_width = np.mean(stereo_width[start_idx:i])
                doubles.append((start_time, end_time, region_amp_db, region_width))

    # Handle case where track ends in a double
    if in_double:
        start_time = times[start_idx]
        end_time = times[-1]
        if end_time - start_time >= min_duration:
            region_amp_db = np.mean(amplitude_db[start_idx:])
            region_width = np.mean(stereo_width[start_idx:])
            doubles.append((start_time, end_time, region_amp_db, region_width))

    return doubles


def classify_double_type(
    amplitude_boost_db: float,
    stereo_width: float,
    spectral_density_increase: float,
    phase_coherence: float
) -> DoubleType:
    """
    Classify the type of double based on audio characteristics.
    """
    # High width + low coherence = panned double takes
    if stereo_width > 0.4 and phase_coherence < 0.7:
        return DoubleType.STACK

    # High width + high coherence = delay/effect
    if stereo_width > 0.3 and phase_coherence > 0.8:
        return DoubleType.DELAY_EFFECT

    # Moderate boost, narrow = single overdub
    if amplitude_boost_db > 2 and stereo_width < 0.2:
        return DoubleType.FULL_DOUBLE

    # Low amplitude, high width = whisper double
    if amplitude_boost_db < 2 and stereo_width > 0.3:
        return DoubleType.WHISPER

    return DoubleType.UNKNOWN


# =============================================================================
# MAIN DETECTION FUNCTION
# =============================================================================

def detect_vocal_doubles(
    audio_path: str,
    use_stem_separation: bool = False,
    bpm: Optional[float] = None,
    syllables: Optional[List[dict]] = None,
    rhyme_groups: Optional[dict] = None,
    sensitivity: str = "normal",  # "low", "normal", "high"
    verbose: bool = True
) -> DoubleDetectionResult:
    """
    Full vocal double detection pipeline.

    Args:
        audio_path: Path to audio file
        use_stem_separation: Whether to run Demucs first (slower but better)
        bpm: Track BPM for bar position calculation
        syllables: List of syllables with timing for word alignment
        rhyme_groups: Dict of rhyme groups for is_rhyme_word detection
        sensitivity: Detection sensitivity ("low", "normal", "high")
        verbose: Print progress

    Returns:
        DoubleDetectionResult with all detected doubles
    """
    # Sensitivity presets
    sensitivity_presets = {
        "low": {"amplitude_threshold_db": 6.0, "width_threshold": 0.35, "require_both": True},
        "normal": {"amplitude_threshold_db": 4.5, "width_threshold": 0.25, "require_both": False},
        "high": {"amplitude_threshold_db": 3.0, "width_threshold": 0.15, "require_both": False},
    }
    sens_params = sensitivity_presets.get(sensitivity, sensitivity_presets["normal"])

    if verbose:
        print(f"  Sensitivity: {sensitivity}")
        print("  Loading audio...")

    # Optionally separate vocals first
    if use_stem_separation:
        if verbose:
            print("  Separating vocals with Demucs (this takes a few minutes)...")
        try:
            audio_path = separate_vocals_demucs(audio_path)
            if verbose:
                print(f"  Vocals saved to: {audio_path}")
        except Exception as e:
            if verbose:
                print(f"  Warning: Demucs failed ({e}), using full mix")

    # Load audio in stereo
    y, sr = load_audio_stereo(audio_path)
    duration = len(y[0] if y.ndim == 2 else y) / sr

    if verbose:
        print(f"  Duration: {duration:.1f}s, Stereo: {y.ndim == 2}")

    # Compute features
    if verbose:
        print("  Computing amplitude envelope...")
    times, amplitude = compute_amplitude_envelope(y, sr)

    if verbose:
        print("  Computing stereo width...")
    _, stereo_width = compute_stereo_width(y, sr)

    # Ensure same length
    min_len = min(len(times), len(amplitude), len(stereo_width))
    times = times[:min_len]
    amplitude = amplitude[:min_len]
    stereo_width = stereo_width[:min_len]

    if verbose:
        print("  Computing spectral features...")
    _, centroid, bandwidth, flatness = compute_spectral_density(y, sr)
    centroid = centroid[:min_len]
    bandwidth = bandwidth[:min_len]
    flatness = flatness[:min_len]

    if verbose:
        print("  Computing phase coherence...")
    _, phase_coh = compute_phase_coherence(y, sr)
    phase_coh = phase_coh[:min_len] if len(phase_coh) >= min_len else np.ones(min_len)

    # Estimate baseline
    if verbose:
        print("  Estimating vocal baseline...")
    baseline = estimate_baseline(
        times, amplitude, stereo_width, centroid, bandwidth, flatness
    )

    if verbose:
        print(f"    Baseline amplitude: {baseline.mean_amplitude:.4f}")
        print(f"    Baseline stereo width: {baseline.mean_stereo_width:.3f}")

    # Detect doubles
    if verbose:
        print("  Detecting doubled sections...")
    raw_doubles = detect_doubles_simple(
        times, amplitude, stereo_width, baseline,
        amplitude_threshold_db=sens_params["amplitude_threshold_db"],
        width_threshold=sens_params["width_threshold"],
        require_both=sens_params["require_both"]
    )

    if verbose:
        print(f"    Found {len(raw_doubles)} potential doubles")

    # Convert to VocalDouble objects with additional analysis
    doubles = []
    bar_duration = (60 / bpm) * 4 if bpm else None

    for start, end, amp_boost, width in raw_doubles:
        # Find corresponding indices
        start_idx = np.searchsorted(times, start)
        end_idx = np.searchsorted(times, end)

        # Get spectral density for this region
        region_centroid = np.mean(centroid[start_idx:end_idx])
        region_bandwidth = np.mean(bandwidth[start_idx:end_idx])
        spectral_increase = (region_bandwidth - baseline.mean_spectral_bandwidth) / (baseline.mean_spectral_bandwidth + 1)

        # Get phase coherence
        region_phase = np.mean(phase_coh[start_idx:end_idx]) if start_idx < len(phase_coh) else 1.0

        # Classify type
        double_type = classify_double_type(amp_boost, width, spectral_increase, region_phase)

        # Calculate bar position
        if bar_duration:
            position_in_bar = (start % bar_duration) / bar_duration * 4
        else:
            position_in_bar = 0.0

        # Find matching word/phrase
        word_or_phrase = ""
        is_rhyme = False
        is_line_end = False

        if syllables:
            matching_syls = [
                s for s in syllables
                if s.get('start', 0) >= start - 0.1 and s.get('end', 0) <= end + 0.1
            ]
            if matching_syls:
                word_or_phrase = " ".join(s.get('text', '') for s in matching_syls)

                # Check if any are rhyme words
                if rhyme_groups:
                    for syl in matching_syls:
                        syl_id = syl.get('id', '')
                        for group_ids in rhyme_groups.values():
                            if syl_id in group_ids:
                                is_rhyme = True
                                break

                # Check if line ending (last syllable has is_word_end and gap after)
                if matching_syls:
                    last_syl = matching_syls[-1]
                    if last_syl.get('is_word_end', False):
                        # Check for gap after
                        next_syls = [s for s in syllables if s.get('start', 0) > last_syl.get('end', 0)]
                        if next_syls:
                            gap = next_syls[0].get('start', 0) - last_syl.get('end', 0)
                            is_line_end = gap > 0.3  # 300ms gap suggests line ending

        # Calculate confidence based on how far above threshold
        confidence = min(1.0, (amp_boost / 6.0 + width / 0.4) / 2)

        doubles.append(VocalDouble(
            start_time=start,
            end_time=end,
            word_or_phrase=word_or_phrase.strip(),
            confidence=confidence,
            double_type=double_type,
            stereo_width=width,
            amplitude_boost_db=amp_boost,
            spectral_density=spectral_increase,
            position_in_bar=position_in_bar,
            is_rhyme_word=is_rhyme,
            is_line_ending=is_line_end
        ))

    # Calculate summary stats
    total_doubled_time = sum(d.duration for d in doubles)
    doubled_ratio = total_doubled_time / duration if duration > 0 else 0

    # Most common type
    if doubles:
        type_counts = {}
        for d in doubles:
            type_counts[d.double_type] = type_counts.get(d.double_type, 0) + 1
        most_common = max(type_counts, key=type_counts.get)
    else:
        most_common = DoubleType.UNKNOWN

    # Doubles per bar
    num_bars = duration / bar_duration if bar_duration else 0
    doubles_per_bar = len(doubles) / num_bars if num_bars > 0 else 0

    if verbose:
        print(f"\n  Results:")
        print(f"    Doubles found: {len(doubles)}")
        print(f"    Total doubled time: {total_doubled_time:.1f}s ({doubled_ratio*100:.1f}%)")
        print(f"    Doubles per bar: {doubles_per_bar:.2f}")
        print(f"    Most common type: {most_common.value}")

    return DoubleDetectionResult(
        doubles=doubles,
        baseline=baseline,
        total_doubled_time=total_doubled_time,
        doubled_ratio=doubled_ratio,
        double_density_per_bar=doubles_per_bar,
        most_common_type=most_common
    )


def print_doubles_report(result: DoubleDetectionResult, show_all: bool = False):
    """Print a formatted report of detected doubles."""
    print("\n" + "=" * 60)
    print("VOCAL DOUBLES REPORT")
    print("=" * 60)

    print(f"""
SUMMARY
  Total doubles: {len(result.doubles)}
  Doubled time: {result.total_doubled_time:.1f}s ({result.doubled_ratio*100:.1f}% of track)
  Per bar: {result.double_density_per_bar:.2f}
  Most common type: {result.most_common_type.value}

BASELINE (single voice)
  Amplitude: {result.baseline.mean_amplitude:.4f} (+/- {result.baseline.std_amplitude:.4f})
  Stereo width: {result.baseline.mean_stereo_width:.3f} (+/- {result.baseline.std_stereo_width:.3f})
""")

    # Group by type
    by_type = {}
    for d in result.doubles:
        t = d.double_type.value
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(d)

    print("BY TYPE:")
    for dtype, doubles in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"  {dtype}: {len(doubles)}")

    # Show rhyme words that are doubled
    rhyme_doubles = [d for d in result.doubles if d.is_rhyme_word]
    if rhyme_doubles:
        print(f"\nRHYME WORDS DOUBLED: {len(rhyme_doubles)}")
        for d in rhyme_doubles[:10]:
            print(f"  [{d.start_time:.2f}s] \"{d.word_or_phrase}\" (+{d.amplitude_boost_db:.1f}dB, width={d.stereo_width:.2f})")

    # Show line endings doubled
    line_end_doubles = [d for d in result.doubles if d.is_line_ending]
    if line_end_doubles:
        print(f"\nLINE ENDINGS DOUBLED: {len(line_end_doubles)}")
        for d in line_end_doubles[:10]:
            print(f"  [{d.start_time:.2f}s] \"{d.word_or_phrase}\" (+{d.amplitude_boost_db:.1f}dB)")

    if show_all:
        print("\nALL DOUBLES:")
        for d in result.doubles:
            flags = []
            if d.is_rhyme_word:
                flags.append("RHYME")
            if d.is_line_ending:
                flags.append("LINE_END")
            flag_str = f" [{','.join(flags)}]" if flags else ""
            print(f"  [{d.start_time:.2f}-{d.end_time:.2f}s] \"{d.word_or_phrase}\" "
                  f"type={d.double_type.value} +{d.amplitude_boost_db:.1f}dB width={d.stereo_width:.2f}{flag_str}")
