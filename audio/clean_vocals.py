"""
Clean Demucs-separated vocals to remove instrumental bleed/"dirt".

Uses noisereduce library with the instrumental track as a noise profile
to subtract any leaked instrumental from the vocals.
"""

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from pathlib import Path


def clean_vocals_with_instrumental_profile(
    vocals_path: str,
    instrumental_path: str,
    output_path: str = None,
    prop_decrease: float = 0.8,
    n_fft: int = 2048
) -> np.ndarray:
    """
    Clean vocals by using the instrumental as a noise profile.

    This subtracts the spectral characteristics of the instrumental
    from the vocals, removing any "bleed" that Demucs left behind.

    Args:
        vocals_path: Path to Demucs-separated vocals
        instrumental_path: Path to instrumental (bass+drums+other)
        output_path: Where to save cleaned vocals (optional)
        prop_decrease: How aggressively to reduce noise (0-1)
        n_fft: FFT size for spectral analysis

    Returns:
        Cleaned vocal audio array
    """
    # Load audio
    vocals, sr = librosa.load(vocals_path, sr=None, mono=False)
    instrumental, _ = librosa.load(instrumental_path, sr=sr, mono=False)

    # Handle mono/stereo
    if vocals.ndim == 1:
        vocals = vocals[np.newaxis, :]
    if instrumental.ndim == 1:
        instrumental = instrumental[np.newaxis, :]

    # Match lengths
    min_len = min(vocals.shape[1], instrumental.shape[1])
    vocals = vocals[:, :min_len]
    instrumental = instrumental[:, :min_len]

    print(f"Vocals shape: {vocals.shape}")
    print(f"Instrumental shape: {instrumental.shape}")
    print(f"Sample rate: {sr}")

    # Clean each channel
    cleaned_channels = []
    for ch in range(vocals.shape[0]):
        print(f"Cleaning channel {ch+1}/{vocals.shape[0]}...")

        # Use instrumental as noise profile
        cleaned = nr.reduce_noise(
            y=vocals[ch],
            sr=sr,
            y_noise=instrumental[ch],  # The "noise" we want to remove
            prop_decrease=prop_decrease,
            n_fft=n_fft,
            stationary=False  # Non-stationary for music
        )
        cleaned_channels.append(cleaned)

    cleaned_vocals = np.array(cleaned_channels)

    # Save if output path provided
    if output_path:
        # Transpose for soundfile (expects [samples, channels])
        sf.write(output_path, cleaned_vocals.T, sr)
        print(f"Saved cleaned vocals to: {output_path}")

    return cleaned_vocals, sr


def clean_vocals_simple(
    vocals_path: str,
    output_path: str = None,
    prop_decrease: float = 0.5
) -> np.ndarray:
    """
    Simple noise reduction without instrumental profile.
    Uses stationary noise estimation.
    """
    vocals, sr = librosa.load(vocals_path, sr=None, mono=False)

    if vocals.ndim == 1:
        vocals = vocals[np.newaxis, :]

    cleaned_channels = []
    for ch in range(vocals.shape[0]):
        cleaned = nr.reduce_noise(
            y=vocals[ch],
            sr=sr,
            prop_decrease=prop_decrease,
            stationary=True
        )
        cleaned_channels.append(cleaned)

    cleaned_vocals = np.array(cleaned_channels)

    if output_path:
        sf.write(output_path, cleaned_vocals.T, sr)
        print(f"Saved cleaned vocals to: {output_path}")

    return cleaned_vocals, sr


if __name__ == "__main__":
    # Test with our Drake track
    data_dir = Path("/home/v12god/rap-notation/data")

    vocals_path = data_dir / "drake_vocals_isolated.wav"
    instrumental_path = data_dir / "stems" / "instrumental.wav"

    if vocals_path.exists() and instrumental_path.exists():
        output_path = data_dir / "drake_vocals_cleaned.wav"

        print("Cleaning vocals using instrumental as noise profile...")
        cleaned, sr = clean_vocals_with_instrumental_profile(
            str(vocals_path),
            str(instrumental_path),
            str(output_path),
            prop_decrease=0.6  # Conservative - don't over-clean
        )

        print(f"\nCleaned vocals shape: {cleaned.shape}")
        print(f"Output saved to: {output_path}")
    else:
        print(f"Missing files:")
        print(f"  Vocals: {vocals_path.exists()}")
        print(f"  Instrumental: {instrumental_path.exists()}")
