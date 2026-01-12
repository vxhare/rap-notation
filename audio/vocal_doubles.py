"""
Vocal Double Detection API

Detects when a rapper doubles/stacks vocals using Mid-Side correlation analysis.
Filters out beat vocals (vocal samples in the production) from true doubles.

Usage:
    from audio.vocal_doubles import VocalDoubleDetector

    detector = VocalDoubleDetector()
    results = detector.analyze("track.wav", word_timestamps=whisper_words)

    for double in results.doubles:
        print(f"{double.word} at {double.start_time:.2f}s (confidence: {double.confidence:.2f})")
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import librosa


@dataclass
class VocalDouble:
    """A detected vocal double."""
    word: str
    start_time: float
    end_time: float
    confidence: float  # 0-1, based on mid-side correlation
    mid_side_correlation: float
    instrumental_bleed: float  # Negative = clean, positive = potential artifact
    is_clean: bool  # Low bleed, high confidence


@dataclass
class VocalDoubleResults:
    """Results from vocal double analysis."""
    doubles: list[VocalDouble]
    beat_vocals: list[dict]  # Segments detected as beat vocals, not rapper doubles
    total_words: int
    doubled_word_count: int
    doubled_percentage: float
    most_doubled_words: list[tuple[str, int]]  # (word, count) sorted by frequency


class VocalDoubleDetector:
    """
    Detect vocal doubles using Mid-Side correlation analysis.

    True doubles: The Side channel (stereo width) "shadows" the Mid channel
                  because doubled vocals are panned slightly differently.
                  High correlation = true double.

    Beat vocals: Vocal samples in the beat have independent rhythm from
                 the rapper's delivery. Low/negative correlation = beat vocal.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.65,
        bleed_threshold: float = 0.5,
        min_word_duration: float = 0.1
    ):
        """
        Args:
            correlation_threshold: Min mid-side correlation for double detection
            bleed_threshold: Max instrumental bleed for "clean" classification
            min_word_duration: Skip words shorter than this (seconds)
        """
        self.correlation_threshold = correlation_threshold
        self.bleed_threshold = bleed_threshold
        self.min_word_duration = min_word_duration

    def analyze(
        self,
        audio_path: str,
        word_timestamps: list[dict],
        instrumental_path: Optional[str] = None
    ) -> VocalDoubleResults:
        """
        Analyze audio for vocal doubles at word level.

        Args:
            audio_path: Path to vocal audio (stereo required)
            word_timestamps: List of {"word": str, "start": float, "end": float}
            instrumental_path: Optional path to instrumental for bleed detection

        Returns:
            VocalDoubleResults with detected doubles
        """
        # Load audio (stereo)
        y, sr = librosa.load(audio_path, sr=None, mono=False)

        if y.ndim == 1:
            raise ValueError("Stereo audio required for mid-side analysis")

        # Load instrumental if provided
        instrumental = None
        if instrumental_path:
            instrumental, _ = librosa.load(instrumental_path, sr=sr, mono=False)
            if instrumental.ndim == 1:
                instrumental = np.stack([instrumental, instrumental])

        # Mid-Side decomposition
        mid = (y[0] + y[1]) / 2
        side = (y[0] - y[1]) / 2

        # Instrumental mid-side if available
        inst_side = None
        if instrumental is not None:
            min_len = min(y.shape[1], instrumental.shape[1])
            inst_side = (instrumental[0, :min_len] - instrumental[1, :min_len]) / 2

        # Analyze each word
        doubles = []
        beat_vocals = []
        word_double_counts = {}

        for word_info in word_timestamps:
            word = word_info.get("word", "")
            start = word_info.get("start", 0)
            end = word_info.get("end", 0)

            duration = end - start
            if duration < self.min_word_duration:
                continue

            # Extract segment
            start_sample = int(start * sr)
            end_sample = int(end * sr)

            if end_sample > len(mid):
                continue

            mid_seg = mid[start_sample:end_sample]
            side_seg = side[start_sample:end_sample]

            # Calculate mid-side correlation
            ms_corr = self._envelope_correlation(mid_seg, side_seg, sr)

            # Calculate instrumental bleed if available
            bleed = 0.0
            if inst_side is not None and end_sample <= len(inst_side):
                inst_seg = inst_side[start_sample:end_sample]
                bleed = self._envelope_correlation(side_seg, inst_seg, sr)

            # Classify
            if ms_corr >= self.correlation_threshold:
                is_clean = bleed < self.bleed_threshold

                double = VocalDouble(
                    word=word.strip(),
                    start_time=start,
                    end_time=end,
                    confidence=min(ms_corr, 1.0),
                    mid_side_correlation=ms_corr,
                    instrumental_bleed=bleed,
                    is_clean=is_clean
                )
                doubles.append(double)

                # Count word frequency
                clean_word = word.strip().lower()
                word_double_counts[clean_word] = word_double_counts.get(clean_word, 0) + 1

            elif ms_corr < 0.3:
                # Likely beat vocal
                beat_vocals.append({
                    "start": start,
                    "end": end,
                    "correlation": ms_corr
                })

        # Sort by frequency
        most_doubled = sorted(
            word_double_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        total_words = len(word_timestamps)
        doubled_count = len(doubles)

        return VocalDoubleResults(
            doubles=doubles,
            beat_vocals=beat_vocals,
            total_words=total_words,
            doubled_word_count=doubled_count,
            doubled_percentage=(doubled_count / total_words * 100) if total_words > 0 else 0,
            most_doubled_words=most_doubled
        )

    def _envelope_correlation(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        sr: int,
        hop_length: int = 512
    ) -> float:
        """Calculate correlation between amplitude envelopes."""
        # Get envelopes
        env1 = np.abs(librosa.util.frame(signal1, frame_length=hop_length, hop_length=hop_length//2)).mean(axis=0)
        env2 = np.abs(librosa.util.frame(signal2, frame_length=hop_length, hop_length=hop_length//2)).mean(axis=0)

        # Match lengths
        min_len = min(len(env1), len(env2))
        if min_len < 2:
            return 0.0

        env1 = env1[:min_len]
        env2 = env2[:min_len]

        # Normalize
        env1 = (env1 - env1.mean()) / (env1.std() + 1e-8)
        env2 = (env2 - env2.mean()) / (env2.std() + 1e-8)

        # Correlation
        return float(np.corrcoef(env1, env2)[0, 1])

    def get_clean_doubles(self, results: VocalDoubleResults) -> list[VocalDouble]:
        """Filter to only clean doubles (low instrumental bleed)."""
        return [d for d in results.doubles if d.is_clean]

    def export_report(self, results: VocalDoubleResults) -> dict:
        """Export results as JSON-serializable dict."""
        return {
            "summary": {
                "total_words": results.total_words,
                "doubled_words": results.doubled_word_count,
                "doubled_percentage": round(results.doubled_percentage, 1),
                "beat_vocal_segments": len(results.beat_vocals),
                "most_doubled": results.most_doubled_words
            },
            "doubles": [
                {
                    "word": d.word,
                    "start": round(d.start_time, 3),
                    "end": round(d.end_time, 3),
                    "confidence": round(d.confidence, 3),
                    "is_clean": d.is_clean,
                    "bleed": round(d.instrumental_bleed, 3)
                }
                for d in results.doubles
            ]
        }


# Convenience function
def detect_doubles(
    audio_path: str,
    word_timestamps: list[dict],
    instrumental_path: Optional[str] = None
) -> VocalDoubleResults:
    """
    Quick function to detect vocal doubles.

    Args:
        audio_path: Path to stereo vocal audio
        word_timestamps: List of {"word": str, "start": float, "end": float}
        instrumental_path: Optional instrumental for bleed filtering

    Returns:
        VocalDoubleResults
    """
    detector = VocalDoubleDetector()
    return detector.analyze(audio_path, word_timestamps, instrumental_path)
