"""
Reverb Detection and Compensation Module

Estimates reverb amount in vocals and adjusts analysis accordingly:
- Detects reverb level (dry to wet)
- Adjusts onset detection sensitivity
- Provides confidence penalties for high-reverb tracks
- Estimates reverb decay time (RT60)
"""

import numpy as np
import librosa
from scipy import signal
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ReverbProfile:
    """Reverb characteristics of a track"""
    reverb_amount: float  # 0 (dry) to 1 (very wet)
    estimated_rt60: float  # Reverb decay time in seconds
    clarity: float  # 0 (muddy) to 1 (clear transients)
    style: str  # "dry", "room", "hall", "plate", "cloud"

    # Recommended adjustments
    onset_threshold_boost: float  # How much to raise onset threshold
    confidence_penalty: float  # How much to reduce confidence scores


def estimate_reverb_amount(
    audio_path: str,
    segment_duration: float = 5.0  # Analyze first N seconds
) -> ReverbProfile:
    """
    Estimate reverb amount from audio characteristics.

    Uses multiple indicators:
    1. Transient sharpness (reverb smooths transients)
    2. Spectral decay rate
    3. Energy envelope decay
    4. Correlation between successive frames
    """
    y, sr = librosa.load(audio_path, duration=segment_duration, mono=True)

    # 1. Transient sharpness via onset strength variance
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_variance = np.var(onset_env) / (np.mean(onset_env) + 1e-6)

    # High variance = sharp transients = dry
    # Low variance = smoothed = reverby
    transient_score = min(1.0, onset_variance / 2.0)

    # 2. Spectral flux decay
    # Reverb causes spectral energy to decay slowly
    S = np.abs(librosa.stft(y))
    spectral_flux = np.diff(S, axis=1)
    flux_decay = np.mean(np.abs(spectral_flux[:, 1:] / (spectral_flux[:, :-1] + 1e-6)))

    # 3. Autocorrelation (reverb increases correlation)
    # Look at correlation at reverb-typical delays (50-200ms)
    autocorr = np.correlate(y[:sr], y[:sr], mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Positive lags only

    # Check correlation at typical reverb delays
    delay_50ms = int(0.05 * sr)
    delay_200ms = int(0.2 * sr)

    if delay_200ms < len(autocorr):
        reverb_correlation = np.mean(autocorr[delay_50ms:delay_200ms]) / (autocorr[0] + 1e-6)
    else:
        reverb_correlation = 0.0

    # 4. Energy envelope analysis
    # Compute RMS envelope
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Look for decay tails after peaks
    peaks, _ = signal.find_peaks(rms, height=np.mean(rms))
    decay_rates = []

    for peak in peaks[:10]:  # Analyze first 10 peaks
        if peak + 20 < len(rms):
            # Measure decay over next 200ms
            decay_segment = rms[peak:peak+20]
            if decay_segment[0] > 0:
                decay_rate = (decay_segment[0] - decay_segment[-1]) / decay_segment[0]
                decay_rates.append(decay_rate)

    avg_decay_rate = np.mean(decay_rates) if decay_rates else 0.5

    # Combine indicators
    # Low transient score = reverby
    # High reverb correlation = reverby
    # Low decay rate = reverby (sound lingers)

    reverb_indicators = [
        1.0 - transient_score,  # Inverted
        reverb_correlation * 5,  # Scale up
        1.0 - avg_decay_rate,  # Inverted
    ]

    reverb_amount = np.clip(np.mean(reverb_indicators), 0, 1)

    # Estimate RT60 (very rough)
    # Based on decay rate
    if avg_decay_rate > 0:
        estimated_rt60 = 0.2 / avg_decay_rate  # Rough estimate
    else:
        estimated_rt60 = 2.0  # Long reverb

    estimated_rt60 = np.clip(estimated_rt60, 0.1, 3.0)

    # Clarity (inverse of reverb amount with more weight on transients)
    clarity = transient_score * 0.7 + avg_decay_rate * 0.3

    # Classify style
    if reverb_amount < 0.2:
        style = "dry"
    elif reverb_amount < 0.4:
        style = "room"
    elif reverb_amount < 0.6:
        style = "hall"
    elif reverb_amount < 0.8:
        style = "plate"
    else:
        style = "cloud"  # Very wet, characteristic of cloud rap

    # Calculate adjustments
    # More reverb = need higher onset threshold
    onset_threshold_boost = reverb_amount * 0.3  # Up to 30% boost

    # More reverb = less confidence in timing
    confidence_penalty = reverb_amount * 0.2  # Up to 20% penalty

    return ReverbProfile(
        reverb_amount=reverb_amount,
        estimated_rt60=estimated_rt60,
        clarity=clarity,
        style=style,
        onset_threshold_boost=onset_threshold_boost,
        confidence_penalty=confidence_penalty
    )


def get_adjusted_onset_params(reverb_profile: ReverbProfile) -> dict:
    """
    Get onset detection parameters adjusted for reverb level.
    """
    base_params = {
        'hop_length': 512,
        'n_mels': 128,
        'fmin': 100,
        'fmax': 4000,
        'peak_threshold': 0.3,
        'min_distance_ms': 50,
    }

    # Adjust based on reverb
    if reverb_profile.style == "dry":
        # Very sensitive, can catch subtle attacks
        base_params['peak_threshold'] = 0.2
        base_params['min_distance_ms'] = 40

    elif reverb_profile.style == "room":
        # Slight adjustment
        base_params['peak_threshold'] = 0.25
        base_params['min_distance_ms'] = 50

    elif reverb_profile.style == "hall":
        # Moderate adjustment
        base_params['peak_threshold'] = 0.35
        base_params['min_distance_ms'] = 60

    elif reverb_profile.style == "plate":
        # More aggressive filtering
        base_params['peak_threshold'] = 0.4
        base_params['min_distance_ms'] = 70

    else:  # cloud
        # Very wet - only catch clear attacks
        base_params['peak_threshold'] = 0.5
        base_params['min_distance_ms'] = 80

    return base_params


def detect_onsets_reverb_aware(
    audio_path: str,
    reverb_profile: Optional[ReverbProfile] = None
) -> Tuple[np.ndarray, ReverbProfile]:
    """
    Detect onsets with reverb-aware parameters.

    Returns (onset_times, reverb_profile)
    """
    # Get reverb profile if not provided
    if reverb_profile is None:
        reverb_profile = estimate_reverb_amount(audio_path)

    # Get adjusted parameters
    params = get_adjusted_onset_params(reverb_profile)

    # Load audio
    y, sr = librosa.load(audio_path, mono=True)

    # Compute mel spectrogram focused on vocal range
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=params['n_mels'],
        fmin=params['fmin'],
        fmax=params['fmax'],
        hop_length=params['hop_length']
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # Compute spectral flux
    flux = np.diff(S_db, axis=1)
    flux = np.maximum(0, flux)  # Only positive changes
    onset_strength = np.mean(flux, axis=0)

    # Normalize
    onset_strength = onset_strength / (np.max(onset_strength) + 1e-6)

    # Apply threshold
    threshold = params['peak_threshold']
    min_distance_samples = int(params['min_distance_ms'] * sr / (1000 * params['hop_length']))

    # Find peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(
        onset_strength,
        height=threshold,
        distance=min_distance_samples
    )

    # Convert to time
    onset_times = librosa.frames_to_time(peaks, sr=sr, hop_length=params['hop_length'])

    return onset_times, reverb_profile


def apply_confidence_adjustment(
    syllables: list,
    reverb_profile: ReverbProfile
) -> list:
    """
    Apply confidence penalty to syllables based on reverb.
    """
    penalty = reverb_profile.confidence_penalty

    for syl in syllables:
        if hasattr(syl, 'confidence'):
            syl.confidence = syl.confidence * (1 - penalty)

    return syllables
