"""
Vocal Double Detection using Mid-Side Analysis

Based on Gemini's approach - much smarter than amplitude/width alone.

Key insight:
- Mid = (L + R) / 2  → Lead vocal (center panned)
- Side = (L - R) / 2 → Stereo width (doubles, reverb, samples)

True doubles "shadow" the lead vocal - when Drake stops, the double stops.
Beat samples have their own rhythm - they keep playing during gaps.

We measure this with envelope correlation between Mid and Side.
"""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SegmentAnalysis:
    """Analysis result for a segment"""
    start_time: float
    end_time: float
    is_double: bool
    confidence: float
    reason: str
    mid_side_correlation: float
    mid_energy: float
    side_energy: float


@dataclass
class MidSideResult:
    """Complete mid-side analysis result"""
    segments: List[SegmentAnalysis]
    true_doubles: List[SegmentAnalysis]
    beat_vocals: List[SegmentAnalysis]
    ambiguous: List[SegmentAnalysis]
    total_double_time: float
    total_beat_vocal_time: float
    double_ratio: float


class VocalDoubleMidSide:
    """
    Detects true vocal doubles using Mid-Side envelope correlation.
    """

    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sr = sample_rate
        self.hop_length = hop_length

    def analyze(
        self,
        audio_path: str,
        chunk_duration: float = 0.5,  # Analyze in smaller chunks for more detail
        correlation_threshold_high: float = 0.65,
        correlation_threshold_low: float = 0.3,
        silence_threshold: float = 0.01,
        verbose: bool = True
    ) -> MidSideResult:
        """
        Analyze a stereo vocal stem to distinguish true doubles from beat vocals.

        Args:
            audio_path: Path to stereo audio file
            chunk_duration: Analysis window size in seconds
            correlation_threshold_high: Above this = true double
            correlation_threshold_low: Below this = beat vocal
            silence_threshold: Ignore chunks below this energy
            verbose: Print progress

        Returns:
            MidSideResult with classified segments
        """
        if verbose:
            print("  Loading audio (stereo)...")

        # Load stereo
        y, _ = librosa.load(audio_path, sr=self.sr, mono=False)

        if y.ndim < 2:
            raise ValueError("Input must be stereo for Mid-Side analysis")

        duration = y.shape[1] / self.sr

        if verbose:
            print(f"  Duration: {duration:.1f}s")
            print("  Computing Mid-Side channels...")

        # Compute Mid and Side
        mid = (y[0] + y[1]) / 2
        side = (y[0] - y[1]) / 2

        # Get envelopes
        mid_env = self._get_envelope(mid)
        side_env = self._get_envelope(side)

        if verbose:
            print(f"  Analyzing in {chunk_duration}s chunks...")

        # Analyze in chunks
        chunk_frames = int(chunk_duration * self.sr / self.hop_length)
        total_frames = len(mid_env)

        segments = []

        for i in range(0, total_frames, chunk_frames):
            end_frame = min(i + chunk_frames, total_frames)

            m_seg = mid_env[i:end_frame]
            s_seg = side_env[i:end_frame]

            # Skip silence
            mid_energy = np.mean(m_seg)
            if mid_energy < silence_threshold:
                continue

            side_energy = np.mean(s_seg)

            # Compute correlation
            if len(m_seg) > 1 and np.std(m_seg) > 0 and np.std(s_seg) > 0:
                correlation = np.corrcoef(m_seg, s_seg)[0, 1]
            else:
                correlation = 0.0

            # Handle NaN
            if np.isnan(correlation):
                correlation = 0.0

            start_time = librosa.frames_to_time(i, sr=self.sr, hop_length=self.hop_length)
            end_time = librosa.frames_to_time(end_frame, sr=self.sr, hop_length=self.hop_length)

            # Classify
            if correlation > correlation_threshold_high:
                is_double = True
                reason = "High Mid/Side correlation (shadowing)"
                confidence = correlation
            elif correlation < correlation_threshold_low:
                is_double = False
                reason = "Low Mid/Side correlation (independent rhythm)"
                confidence = 1 - correlation
            else:
                is_double = None  # Ambiguous
                reason = "Ambiguous correlation"
                confidence = 0.5

            segments.append(SegmentAnalysis(
                start_time=start_time,
                end_time=end_time,
                is_double=is_double,
                confidence=confidence,
                reason=reason,
                mid_side_correlation=correlation,
                mid_energy=mid_energy,
                side_energy=side_energy
            ))

        # Categorize
        true_doubles = [s for s in segments if s.is_double == True]
        beat_vocals = [s for s in segments if s.is_double == False]
        ambiguous = [s for s in segments if s.is_double is None]

        # Calculate times
        total_double_time = sum(s.end_time - s.start_time for s in true_doubles)
        total_beat_vocal_time = sum(s.end_time - s.start_time for s in beat_vocals)
        double_ratio = total_double_time / duration if duration > 0 else 0

        if verbose:
            print(f"\n  Results:")
            print(f"    True doubles: {len(true_doubles)} segments ({total_double_time:.1f}s)")
            print(f"    Beat vocals: {len(beat_vocals)} segments ({total_beat_vocal_time:.1f}s)")
            print(f"    Ambiguous: {len(ambiguous)} segments")
            print(f"    Double ratio: {double_ratio*100:.1f}%")

        return MidSideResult(
            segments=segments,
            true_doubles=true_doubles,
            beat_vocals=beat_vocals,
            ambiguous=ambiguous,
            total_double_time=total_double_time,
            total_beat_vocal_time=total_beat_vocal_time,
            double_ratio=double_ratio
        )

    def _get_envelope(self, audio: np.ndarray) -> np.ndarray:
        """Get smoothed amplitude envelope."""
        frame_length = 1024
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=self.hop_length
        )[0]

        # Normalize
        if np.max(rms) > 0:
            rms = rms / np.max(rms)

        return rms

    def visualize(
        self,
        audio_path: str,
        result: MidSideResult,
        output_path: Optional[str] = None
    ):
        """Visualize Mid-Side analysis with segment classifications."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for visualization")
            return

        y, _ = librosa.load(audio_path, sr=self.sr, mono=False)

        mid = (y[0] + y[1]) / 2
        side = (y[0] - y[1]) / 2

        mid_env = self._get_envelope(mid)
        side_env = self._get_envelope(side)

        times = librosa.frames_to_time(np.arange(len(mid_env)), sr=self.sr, hop_length=self.hop_length)

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        # Plot Mid envelope
        axes[0].plot(times, mid_env, color='blue', linewidth=0.5)
        axes[0].set_ylabel('Mid (Lead Vocal)')
        axes[0].set_title('Mid-Side Analysis for Vocal Double Detection')

        # Plot Side envelope
        axes[1].plot(times, side_env, color='orange', linewidth=0.5)
        axes[1].set_ylabel('Side (Width)')

        # Highlight segments
        for seg in result.true_doubles:
            axes[1].axvspan(seg.start_time, seg.end_time, color='green', alpha=0.3)
        for seg in result.beat_vocals:
            axes[1].axvspan(seg.start_time, seg.end_time, color='red', alpha=0.3)

        # Plot correlation over time
        correlations = [s.mid_side_correlation for s in result.segments]
        seg_times = [(s.start_time + s.end_time) / 2 for s in result.segments]

        axes[2].scatter(seg_times, correlations, c=['green' if s.is_double else 'red' if s.is_double == False else 'gray' for s in result.segments], s=20)
        axes[2].axhline(y=0.65, color='green', linestyle='--', label='Double threshold')
        axes[2].axhline(y=0.3, color='red', linestyle='--', label='Beat vocal threshold')
        axes[2].set_ylabel('Mid/Side Correlation')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend()
        axes[2].set_ylim(-0.5, 1.0)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Saved visualization to: {output_path}")
        else:
            plt.show()

        plt.close()


def print_midside_report(result: MidSideResult, show_all: bool = False):
    """Print a formatted report of the mid-side analysis."""
    print("\n" + "=" * 60)
    print("MID-SIDE VOCAL DOUBLE ANALYSIS")
    print("=" * 60)

    print(f"""
SUMMARY
  True doubles: {len(result.true_doubles)} segments ({result.total_double_time:.1f}s)
  Beat vocals:  {len(result.beat_vocals)} segments ({result.total_beat_vocal_time:.1f}s)
  Ambiguous:    {len(result.ambiguous)} segments
  Double ratio: {result.double_ratio*100:.1f}%

METHOD
  Mid = (L + R) / 2  → Lead vocal (center)
  Side = (L - R) / 2 → Stereo width

  High correlation (>0.65): Side follows Mid → TRUE DOUBLE
  Low correlation (<0.3): Side independent → BEAT VOCAL
""")

    if result.true_doubles:
        print("TRUE DOUBLES (Side shadows Mid):")
        print("-" * 60)
        for s in sorted(result.true_doubles, key=lambda x: -x.confidence)[:10]:
            print(f"  {s.start_time:6.2f}s - {s.end_time:6.2f}s | corr={s.mid_side_correlation:.2f}")

    if result.beat_vocals:
        print("\nBEAT VOCALS (Side has independent rhythm):")
        print("-" * 60)
        for s in sorted(result.beat_vocals, key=lambda x: x.mid_side_correlation)[:10]:
            print(f"  {s.start_time:6.2f}s - {s.end_time:6.2f}s | corr={s.mid_side_correlation:.2f}")
