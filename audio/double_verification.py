"""
Vocal Double Verification Tools

Tools to verify that detected doubles are actually doubled vocals:
1. Export audio clips of detected doubles
2. Visualize waveform/spectrogram comparison
3. A/B comparison of doubled vs non-doubled sections
4. Generate verification report with timestamps
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from audio.vocal_double_detection import VocalDouble, DoubleDetectionResult


def export_double_clips(
    audio_path: str,
    doubles: List[VocalDouble],
    output_dir: str = "double_clips",
    padding: float = 0.5,  # Add context before/after
    max_clips: int = 20
) -> List[str]:
    """
    Export audio clips of detected doubles for listening verification.

    Returns list of exported file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, mono=False)
    if y.ndim == 1:
        y = np.array([y, y])  # Fake stereo for mono

    duration = y.shape[1] / sr

    exported = []

    # Sort by confidence and take top N
    sorted_doubles = sorted(doubles, key=lambda d: d.confidence, reverse=True)[:max_clips]

    for i, d in enumerate(sorted_doubles):
        # Calculate sample positions with padding
        start_sec = max(0, d.start_time - padding)
        end_sec = min(duration, d.end_time + padding)

        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)

        # Extract clip
        clip = y[:, start_sample:end_sample]

        # Create filename with info
        word_safe = d.word_or_phrase[:20].replace(" ", "_").replace("/", "-") if d.word_or_phrase else "unknown"
        filename = f"double_{i+1:02d}_{d.start_time:.1f}s_{word_safe}_{d.double_type.value}.wav"
        filepath = output_dir / filename

        # Save
        sf.write(str(filepath), clip.T, sr)
        exported.append(str(filepath))

        print(f"  [{i+1}] {d.start_time:.2f}-{d.end_time:.2f}s: \"{d.word_or_phrase[:30]}\" "
              f"(conf={d.confidence:.2f}, width={d.stereo_width:.2f}, +{d.amplitude_boost_db:.1f}dB)")

    return exported


def export_comparison_clips(
    audio_path: str,
    result: DoubleDetectionResult,
    output_dir: str = "comparison_clips",
    num_pairs: int = 5
) -> List[Tuple[str, str]]:
    """
    Export pairs of clips: doubled section + nearby non-doubled section.
    This helps verify by A/B listening comparison.

    Returns list of (doubled_path, non_doubled_path) tuples.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    y, sr = librosa.load(audio_path, sr=22050, mono=False)
    if y.ndim == 1:
        y = np.array([y, y])

    duration = y.shape[1] / sr

    # Get doubled time ranges
    doubled_ranges = [(d.start_time, d.end_time) for d in result.doubles]

    # Find non-doubled ranges (gaps between doubles)
    non_doubled_ranges = []
    sorted_ranges = sorted(doubled_ranges)

    for i in range(len(sorted_ranges) - 1):
        gap_start = sorted_ranges[i][1]
        gap_end = sorted_ranges[i + 1][0]
        gap_duration = gap_end - gap_start

        if gap_duration > 0.5:  # Need at least 0.5s gap
            non_doubled_ranges.append((gap_start + 0.1, gap_end - 0.1))

    pairs = []

    # Sort doubles by confidence
    sorted_doubles = sorted(result.doubles, key=lambda d: d.confidence, reverse=True)

    for i, d in enumerate(sorted_doubles[:num_pairs]):
        if i >= len(non_doubled_ranges):
            break

        # Export doubled clip
        d_start = int(d.start_time * sr)
        d_end = int(d.end_time * sr)
        doubled_clip = y[:, d_start:d_end]

        doubled_path = output_dir / f"pair{i+1}_A_doubled_{d.start_time:.1f}s.wav"
        sf.write(str(doubled_path), doubled_clip.T, sr)

        # Export non-doubled clip (similar duration)
        target_duration = d.end_time - d.start_time
        nd_start, nd_end = non_doubled_ranges[i]
        nd_duration = nd_end - nd_start

        # Center the clip
        if nd_duration > target_duration:
            center = (nd_start + nd_end) / 2
            nd_start = center - target_duration / 2
            nd_end = center + target_duration / 2

        nd_start_sample = int(nd_start * sr)
        nd_end_sample = int(nd_end * sr)
        non_doubled_clip = y[:, nd_start_sample:nd_end_sample]

        non_doubled_path = output_dir / f"pair{i+1}_B_single_{nd_start:.1f}s.wav"
        sf.write(str(non_doubled_path), non_doubled_clip.T, sr)

        pairs.append((str(doubled_path), str(non_doubled_path)))

        print(f"  Pair {i+1}:")
        print(f"    A (doubled):    {d.start_time:.2f}s - width={d.stereo_width:.2f}, +{d.amplitude_boost_db:.1f}dB")
        print(f"    B (single):     {nd_start:.2f}s - baseline")

    return pairs


def visualize_double(
    audio_path: str,
    double: VocalDouble,
    output_path: Optional[str] = None,
    padding: float = 1.0
):
    """
    Create visualization comparing doubled section to surrounding audio.
    Shows waveform (L/R), amplitude envelope, and stereo width.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization. Install with: pip install matplotlib")
        return

    y, sr = librosa.load(audio_path, sr=22050, mono=False)
    if y.ndim == 1:
        y = np.array([y, y])

    # Get region around the double
    start_sec = max(0, double.start_time - padding)
    end_sec = min(y.shape[1] / sr, double.end_time + padding)

    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)

    region = y[:, start_sample:end_sample]
    times = np.linspace(start_sec, end_sec, region.shape[1])

    # Calculate features
    left = region[0]
    right = region[1]
    mono = (left + right) / 2

    # RMS envelope
    frame_length = 512
    hop_length = 128
    rms = librosa.feature.rms(y=mono, frame_length=frame_length, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length) + start_sec

    # Stereo width (frame by frame)
    n_frames = len(mono) // hop_length
    width = []
    for i in range(n_frames):
        s = i * hop_length
        e = s + frame_length
        if e > len(left):
            break
        l_frame = left[s:e]
        r_frame = right[s:e]
        if np.std(l_frame) > 0 and np.std(r_frame) > 0:
            corr = np.corrcoef(l_frame, r_frame)[0, 1]
            width.append(1 - max(0, corr))
        else:
            width.append(0)
    width = np.array(width)
    width_times = librosa.frames_to_time(np.arange(len(width)), sr=sr, hop_length=hop_length) + start_sec

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Waveform - Left
    axes[0].plot(times, left, color='blue', alpha=0.7, linewidth=0.5)
    axes[0].axvspan(double.start_time, double.end_time, color='red', alpha=0.2, label='Detected double')
    axes[0].set_ylabel('Left Channel')
    axes[0].set_title(f'Vocal Double: "{double.word_or_phrase[:40]}" ({double.double_type.value})')
    axes[0].legend()

    # Waveform - Right
    axes[1].plot(times, right, color='green', alpha=0.7, linewidth=0.5)
    axes[1].axvspan(double.start_time, double.end_time, color='red', alpha=0.2)
    axes[1].set_ylabel('Right Channel')

    # Amplitude envelope
    axes[2].plot(rms_times[:len(rms)], rms, color='purple', linewidth=1)
    axes[2].axvspan(double.start_time, double.end_time, color='red', alpha=0.2)
    axes[2].set_ylabel('Amplitude (RMS)')
    axes[2].axhline(y=np.mean(rms), color='gray', linestyle='--', alpha=0.5, label='Mean')
    axes[2].legend()

    # Stereo width
    axes[3].plot(width_times[:len(width)], width, color='orange', linewidth=1)
    axes[3].axvspan(double.start_time, double.end_time, color='red', alpha=0.2)
    axes[3].set_ylabel('Stereo Width')
    axes[3].set_xlabel('Time (s)')
    axes[3].axhline(y=np.mean(width), color='gray', linestyle='--', alpha=0.5, label='Mean')
    axes[3].legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"  Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()


def generate_verification_report(
    audio_path: str,
    result: DoubleDetectionResult,
    output_path: str = "doubles_report.txt"
) -> str:
    """
    Generate a text report with timestamps for manual verification.
    Format: timestamps you can seek to in any audio player.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("VOCAL DOUBLES VERIFICATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Audio: {audio_path}")
    lines.append(f"Total doubles: {len(result.doubles)}")
    lines.append(f"Doubled time: {result.total_doubled_time:.1f}s ({result.doubled_ratio*100:.1f}%)")
    lines.append("")
    lines.append("TIMESTAMPS TO CHECK (sorted by confidence):")
    lines.append("-" * 70)
    lines.append(f"{'#':<4} {'Time':<12} {'Conf':<6} {'Width':<6} {'dB':<6} {'Type':<12} {'Words'}")
    lines.append("-" * 70)

    # Sort by confidence
    sorted_doubles = sorted(result.doubles, key=lambda d: d.confidence, reverse=True)

    for i, d in enumerate(sorted_doubles, 1):
        time_str = f"{int(d.start_time//60)}:{d.start_time%60:05.2f}"
        words = d.word_or_phrase[:30] if d.word_or_phrase else "-"
        lines.append(f"{i:<4} {time_str:<12} {d.confidence:.2f}   {d.stereo_width:.2f}   "
                    f"{d.amplitude_boost_db:+.1f}  {d.double_type.value:<12} {words}")

    lines.append("")
    lines.append("HOW TO VERIFY:")
    lines.append("1. Open the audio file in any player (VLC, Audacity, etc.)")
    lines.append("2. Seek to each timestamp")
    lines.append("3. Listen for: wider stereo field, thicker sound, louder emphasis")
    lines.append("4. Compare to surrounding 'single' vocals")
    lines.append("")
    lines.append("WHAT DOUBLED VOCALS SOUND LIKE:")
    lines.append("- Fuller/thicker than single voice")
    lines.append("- Often panned left AND right (sounds wide)")
    lines.append("- Slightly louder/more present")
    lines.append("- May have slight timing artifacts (flamming)")

    report = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Report saved to: {output_path}")
    return report


def quick_verify(audio_path: str, result: DoubleDetectionResult, num_samples: int = 5):
    """
    Quick verification: print top N doubles with timestamps for listening.
    """
    print("\n" + "=" * 60)
    print("QUICK VERIFICATION - Top doubles to check")
    print("=" * 60)
    print(f"Open {audio_path} and seek to these times:\n")

    sorted_doubles = sorted(result.doubles, key=lambda d: d.confidence, reverse=True)[:num_samples]

    for i, d in enumerate(sorted_doubles, 1):
        mins = int(d.start_time // 60)
        secs = d.start_time % 60

        print(f"{i}. {mins}:{secs:05.2f} - {mins}:{(d.end_time % 60):05.2f}")
        print(f"   Confidence: {d.confidence:.0%}")
        print(f"   Stereo width: {d.stereo_width:.2f} (0=mono, 1=full stereo)")
        print(f"   Amplitude boost: {d.amplitude_boost_db:+.1f} dB")
        print(f"   Type: {d.double_type.value}")
        if d.word_or_phrase:
            print(f"   Words: \"{d.word_or_phrase[:50]}\"")
        print()

    print("Listen for: wider stereo image, thicker sound, louder than surrounding vocals")
