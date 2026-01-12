"""
Rap Notation Analysis Pipeline

Full audio-to-analysis pipeline:
    audio.mp3 → transcription → beat alignment → rhyme/flow analysis → output

Usage:
    from pipeline import analyze_track
    result = analyze_track("path/to/track.mp3")
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import asdict
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np


def analyze_track(
    audio_path: str,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    whisper_model: str = "base",
    verbose: bool = True
) -> dict:
    """
    Full analysis pipeline for a rap track.

    Args:
        audio_path: Path to audio file (mp3, wav, etc.)
        title: Track title (optional, extracted from filename if not provided)
        artist: Artist name (optional)
        whisper_model: Whisper model size ("tiny", "base", "small", "medium", "large")
        verbose: Print progress

    Returns:
        Complete analysis dict
    """

    start_time = time.time()

    if verbose:
        print("=" * 60)
        print("RAP NOTATION ANALYSIS PIPELINE")
        print("=" * 60)
        print(f"File: {audio_path}")
        print()

    # Extract title from filename if not provided
    if not title:
        title = Path(audio_path).stem

    # =========================================================================
    # STEP 1: LOAD AUDIO & DETECT BEAT
    # =========================================================================
    if verbose:
        print("[1/8] Analyzing beat and tempo...")

    import librosa

    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)

    if verbose:
        print(f"      Duration: {duration:.1f}s")

    # Tempo detection
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # Handle both old and new librosa return types
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    else:
        tempo = float(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if verbose:
        print(f"      Tempo: {tempo:.1f} BPM")
        print(f"      Beats detected: {len(beat_times)}")

    # Estimate bar boundaries (assuming 4/4 time)
    beats_per_bar = 4
    bar_duration = (60 / tempo) * beats_per_bar
    total_bars = int(duration / bar_duration)

    if verbose:
        print(f"      Estimated bars: {total_bars}")

    beat_analysis = {
        "bpm": tempo,
        "beat_times": beat_times.tolist(),
        "total_bars": total_bars,
        "bar_duration": bar_duration,
        "duration_seconds": duration
    }

    # =========================================================================
    # STEP 2: REVERB ANALYSIS
    # =========================================================================
    if verbose:
        print("\n[2/8] Analyzing reverb characteristics...")

    from audio.reverb_detection import estimate_reverb_amount, get_adjusted_onset_params

    reverb_profile = estimate_reverb_amount(audio_path)

    if verbose:
        print(f"      Reverb style: {reverb_profile.style}")
        print(f"      Reverb amount: {reverb_profile.reverb_amount:.2f}")
        print(f"      Clarity: {reverb_profile.clarity:.2f}")
        print(f"      Est. RT60: {reverb_profile.estimated_rt60:.2f}s")

    # =========================================================================
    # STEP 3: TRANSCRIBE WITH WHISPER
    # =========================================================================
    if verbose:
        print(f"\n[3/8] Transcribing with Whisper ({whisper_model})...")

    import whisper

    model = whisper.load_model(whisper_model)

    # Transcribe with word timestamps
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en"
    )

    # Extract words with timestamps
    words = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            words.append({
                "text": word_info["word"].strip(),
                "start": word_info["start"],
                "end": word_info["end"]
            })

    full_text = result.get("text", "")

    if verbose:
        print(f"      Words transcribed: {len(words)}")
        print(f"      Preview: {full_text[:80]}...")

    # =========================================================================
    # STEP 4: ADLIB DETECTION
    # =========================================================================
    if verbose:
        print("\n[4/8] Detecting adlibs...")

    from audio.adlib_detection import detect_adlibs_with_audio, calculate_adlib_density

    # Convert words for adlib detection
    whisper_words = [{"word": w["text"], "start": w["start"], "end": w["end"]} for w in words]

    adlibs, main_words = detect_adlibs_with_audio(audio_path, whisper_words)

    # Convert main_words back to original format
    words = [{"text": w["word"], "start": w["start"], "end": w["end"]} for w in main_words]

    adlib_stats = calculate_adlib_density(adlibs, duration, total_bars)

    if verbose:
        print(f"      Adlibs detected: {len(adlibs)}")
        print(f"      Adlibs/bar: {adlib_stats['adlibs_per_bar']:.2f}")
        if adlib_stats['most_common']:
            common = [f"{w}({c})" for w, c in adlib_stats['most_common'][:3]]
            print(f"      Most common: {', '.join(common)}")
        print(f"      Main vocal words: {len(words)}")

    # =========================================================================
    # STEP 5: DETECT RHYTHMIC SYLLABLES (actual vocal events)
    # =========================================================================
    if verbose:
        print("\n[5/8] Detecting rhythmic syllables...")

    from audio.onset_detection import detect_vocal_onsets_aggressive
    from audio.syllable_alignment import align_words_to_onsets_greedy
    from phonetics.phoneme_lookup import get_phonemes, get_rime, get_stressed_vowel
    from audio.reverb_detection import detect_onsets_reverb_aware

    # Detect actual vocal onsets (rhythmic events) - using reverb-aware detection
    onset_params = get_adjusted_onset_params(reverb_profile)
    onset_times, _ = detect_onsets_reverb_aware(audio_path, reverb_profile)
    onset_times = onset_times.tolist()

    if verbose:
        print(f"      Raw onsets detected: {len(onset_times)}")

    # Align Whisper words to onsets, splitting into syllables
    aligned_syllables = align_words_to_onsets_greedy(words, onset_times, bar_duration)

    if verbose:
        print(f"      Aligned syllables: {len(aligned_syllables)}")

    # Build syllable list with phonemes
    syllables = []

    for syl in aligned_syllables:
        text = syl.text
        text_clean = text.strip(".,!?;:'\"").lower()

        if not text_clean:
            continue

        # Get phonemes for this syllable
        phoneme_variants = get_phonemes(text_clean)
        phonemes = phoneme_variants[0] if phoneme_variants else []

        nucleus, coda = get_rime(phonemes)
        stressed = get_stressed_vowel(phonemes)

        syllables.append({
            "id": f"syl_{syl.index}",
            "text": text,
            "word": syl.word,
            "start": syl.start_sec,
            "end": syl.end_sec,
            "bar": syl.bar,
            "beat": syl.beat,
            "is_word_start": syl.is_word_start,
            "is_word_end": syl.is_word_end,
            "phonemes": phonemes,
            "nucleus": nucleus,
            "coda": coda,
            "stressed_vowel": stressed
        })

    # Count phoneme coverage
    with_phonemes = sum(1 for s in syllables if s["phonemes"])
    coverage = with_phonemes / len(syllables) * 100 if syllables else 0

    if verbose:
        print(f"      Syllables: {len(syllables)}")
        print(f"      Phoneme coverage: {coverage:.1f}%")

    # =========================================================================
    # STEP 6: RHYME DETECTION
    # =========================================================================
    if verbose:
        print("\n[6/8] Detecting rhymes...")

    from analysis.rhyme_detector import RhymeDetector

    detector = RhymeDetector(min_score=0.7)
    rhyme_scheme = detector.analyze(syllables, beats_per_bar=4)

    if verbose:
        print(f"      Rhyme groups: {len(rhyme_scheme.groups)}")
        print(f"      Rhyme density: {rhyme_scheme.rhyme_density:.2f} per bar")

        # Show top rhyme groups
        if rhyme_scheme.groups:
            print("      Top groups:")
            for g in rhyme_scheme.groups[:3]:
                texts = [s["text"] for s in syllables if s["id"] in g.syllable_ids][:5]
                print(f"        [{g.rhyme_type.value}] {', '.join(texts)}")

    # =========================================================================
    # STEP 7: GRID ALIGNMENT & FLOW METRICS
    # =========================================================================
    if verbose:
        print("\n[7/8] Analyzing flow and microtiming...")

    from analysis.grid_alignment import GridAligner
    from analysis.flow_metrics import analyze_flow

    # Grid alignment
    aligner = GridAligner(bpm=tempo)
    aligned = aligner.align_syllables(syllables)
    grid_comparison = aligner.compare_grids(aligned)
    microtiming = aligner.analyze_microtiming(aligned)

    if verbose:
        print(f"      Recommended grid: {grid_comparison['recommended']}")
        print(f"      Microtiming style: {microtiming.style}")
        print(f"      Mean deviation: {microtiming.mean_deviation_ms:.1f}ms")

    # Flow metrics - convert groups to dict format
    rhyme_groups_dict = {
        g.id: g.syllable_ids for g in rhyme_scheme.groups
    }
    flow = analyze_flow(syllables, rhyme_groups_dict)

    if verbose:
        print(f"      Syllables/bar: {flow['density']['syllables_per_bar']:.1f}")
        print(f"      Rhythm complexity: {flow['rhythm']['complexity']:.3f}")

    # =========================================================================
    # STEP 8: PITCH TRACKING & MELODICITY
    # =========================================================================
    if verbose:
        print("\n[8/8] Analyzing pitch and melodicity...")

    from audio.pitch_tracking import analyze_track_melodicity

    melodicity_profile, syllable_pitches = analyze_track_melodicity(
        audio_path,
        syllables=syllables,
        verbose=False
    )

    if verbose:
        print(f"      Overall melodicity: {melodicity_profile.overall_melodicity:.2f}")
        print(f"      Vocal style: {melodicity_profile.style}")
        print(f"      Sung: {melodicity_profile.sung_ratio*100:.0f}%, Rapped: {melodicity_profile.rapped_ratio*100:.0f}%")
        print(f"      Mean pitch: {melodicity_profile.mean_pitch_hz:.0f}Hz")

    # Add pitch data to syllables
    pitch_lookup = {sp.syllable_id: sp for sp in syllable_pitches}
    for syl in syllables:
        sp = pitch_lookup.get(syl['id'])
        if sp:
            syl['melodicity'] = sp.melodicity_score
            syl['pitch_hz'] = sp.mean_f0
            syl['is_sung'] = sp.is_sung

    # =========================================================================
    # STEP 9: VOCAL DOUBLE DETECTION (Mid-Side Analysis)
    # =========================================================================
    if verbose:
        print("\n[9/9] Detecting vocal doubles (Mid-Side analysis)...")

    from audio.vocal_doubles import VocalDoubleDetector

    # Prepare word timestamps for double detection
    word_timestamps = [{"word": w["text"], "start": w["start"], "end": w["end"]} for w in words]

    double_detector = VocalDoubleDetector(
        correlation_threshold=0.65,
        bleed_threshold=0.5
    )

    double_result = double_detector.analyze(
        audio_path,
        word_timestamps=word_timestamps,
        instrumental_path=None  # Would need stems if available
    )

    # Get clean doubles only (low instrumental bleed)
    clean_doubles = double_detector.get_clean_doubles(double_result)

    if verbose:
        print(f"      Total doubled words: {double_result.doubled_word_count}")
        print(f"      Clean doubles: {len(clean_doubles)}")
        print(f"      Percentage doubled: {double_result.doubled_percentage:.1f}%")
        print(f"      Beat vocals filtered: {len(double_result.beat_vocals)}")
        # Show most doubled words
        if double_result.most_doubled_words:
            most = [f'"{w}"({c})' for w, c in double_result.most_doubled_words[:5]]
            print(f"      Most doubled: {', '.join(most)}")

    # =========================================================================
    # STEP 10: GENRE DETECTION (now with real melodicity)
    # =========================================================================
    if verbose:
        print("\n[BONUS] Detecting genre/style...")

    from config.genre_profiles import detect_genre_from_features

    syls_per_bar = len(syllables) / total_bars if total_bars > 0 else 0
    melodicity = melodicity_profile.overall_melodicity

    genre, genre_conf = detect_genre_from_features(
        bpm=tempo,
        syllables_per_bar=syls_per_bar,
        melodicity=melodicity,
        rhyme_density=rhyme_scheme.rhyme_density
    )

    if verbose:
        print(f"      Detected style: {genre.value} (confidence: {genre_conf:.2f})")

    # =========================================================================
    # COMPILE RESULTS
    # =========================================================================
    elapsed = time.time() - start_time

    analysis = {
        "metadata": {
            "title": title,
            "artist": artist,
            "duration_seconds": duration,
            "analyzed_in_seconds": elapsed
        },
        "beat": beat_analysis,
        "transcription": {
            "full_text": full_text,
            "word_count": len(words),
            "syllable_count": len(syllables)
        },
        "syllables": syllables,
        "rhyme_scheme": {
            "groups": [
                {
                    "id": g.id,
                    "type": g.rhyme_type.value,
                    "syllable_ids": g.syllable_ids,
                    "score": g.score
                }
                for g in rhyme_scheme.groups
            ],
            "density": rhyme_scheme.rhyme_density,
            "pattern": rhyme_scheme.end_rhyme_pattern[:50]
        },
        "flow": {
            "grid": grid_comparison,
            "microtiming": {
                "style": microtiming.style,
                "mean_deviation_ms": microtiming.mean_deviation_ms,
                "std_deviation_ms": microtiming.std_deviation_ms
            },
            "metrics": flow
        },
        "genre": {
            "detected": genre.value,
            "confidence": genre_conf
        },
        "reverb": {
            "style": reverb_profile.style,
            "amount": reverb_profile.reverb_amount,
            "clarity": reverb_profile.clarity,
            "estimated_rt60": reverb_profile.estimated_rt60,
            "confidence_penalty": reverb_profile.confidence_penalty
        },
        "adlibs": {
            "total": adlib_stats['total_adlibs'],
            "per_bar": adlib_stats['adlibs_per_bar'],
            "most_common": adlib_stats['most_common'],
            "panning_distribution": adlib_stats['panning_distribution'],
            "detected": [
                {
                    "text": a.text,
                    "start": a.start_time,
                    "end": a.end_time,
                    "confidence": a.confidence,
                    "panning": a.panning,
                    "reason": a.reason
                }
                for a in adlibs
            ]
        },
        "melodicity": {
            "overall": melodicity_profile.overall_melodicity,
            "style": melodicity_profile.style,
            "sung_ratio": melodicity_profile.sung_ratio,
            "rapped_ratio": melodicity_profile.rapped_ratio,
            "mean_pitch_hz": melodicity_profile.mean_pitch_hz,
            "pitch_range_semitones": melodicity_profile.pitch_range_semitones
        },
        "vocal_doubles": {
            "total": double_result.doubled_word_count,
            "clean_doubles": len(clean_doubles),
            "doubled_percentage": double_result.doubled_percentage,
            "beat_vocals_filtered": len(double_result.beat_vocals),
            "most_doubled_words": double_result.most_doubled_words,
            "doubles": [
                {
                    "word": d.word,
                    "start": d.start_time,
                    "end": d.end_time,
                    "confidence": d.confidence,
                    "mid_side_correlation": d.mid_side_correlation,
                    "instrumental_bleed": d.instrumental_bleed,
                    "is_clean": d.is_clean
                }
                for d in double_result.doubles
            ]
        }
    }

    if verbose:
        print()
        print("=" * 60)
        print(f"ANALYSIS COMPLETE ({elapsed:.1f}s)")
        print("=" * 60)

    return analysis


def print_analysis_summary(analysis: dict):
    """Print a readable summary of the analysis."""

    meta = analysis["metadata"]
    beat = analysis["beat"]
    trans = analysis["transcription"]
    rhyme = analysis["rhyme_scheme"]
    flow = analysis["flow"]
    genre = analysis["genre"]

    print()
    print("=" * 60)
    print(f"ANALYSIS: {meta['title']}")
    if meta.get('artist'):
        print(f"Artist: {meta['artist']}")
    print("=" * 60)

    print(f"""
BEAT
  BPM: {beat['bpm']:.1f}
  Duration: {beat['duration_seconds']:.1f}s
  Bars: {beat['total_bars']}

LYRICS
  Words: {trans['word_count']}
  Syllables: {trans['syllable_count']}
  Density: {trans['syllable_count'] / beat['total_bars']:.1f} syl/bar

RHYMES
  Groups: {len(rhyme['groups'])}
  Density: {rhyme['density']:.2f} rhymes/bar
  Pattern: {rhyme['pattern'][:30]}...

FLOW
  Grid: {flow['grid']['recommended']}
  Style: {flow['microtiming']['style']}
  Deviation: {flow['microtiming']['mean_deviation_ms']:.1f}ms
  Complexity: {flow['metrics']['rhythm']['complexity']:.3f}

GENRE
  Detected: {genre['detected']}
  Confidence: {genre['confidence']:.2f}
""")

    # Melodicity section
    mel = analysis.get('melodicity', {})
    if mel:
        print(f"""MELODICITY
  Overall: {mel.get('overall', 0):.2f} (0=rap, 1=singing)
  Style: {mel.get('style', '?')}
  Sung: {mel.get('sung_ratio', 0)*100:.0f}% | Rapped: {mel.get('rapped_ratio', 0)*100:.0f}%
  Mean pitch: {mel.get('mean_pitch_hz', 0):.0f}Hz
""")

    # Reverb section
    rev = analysis.get('reverb', {})
    if rev:
        print(f"""REVERB
  Style: {rev.get('style', '?')}
  Amount: {rev.get('amount', 0):.2f} (0=dry, 1=wet)
  Clarity: {rev.get('clarity', 0):.2f}
  Est. RT60: {rev.get('estimated_rt60', 0):.2f}s
""")

    # Adlib section
    adlib = analysis.get('adlibs', {})
    if adlib:
        common = adlib.get('most_common', [])
        common_str = ', '.join([f"{w}({c})" for w, c in common[:5]]) if common else "none"
        print(f"""ADLIBS
  Total: {adlib.get('total', 0)}
  Per bar: {adlib.get('per_bar', 0):.2f}
  Common: {common_str}
""")

    # Vocal doubles section
    doubles = analysis.get('vocal_doubles', {})
    if doubles:
        most_doubled = doubles.get('most_doubled_words', [])
        words_str = ', '.join([f'"{w}"({c})' for w, c in most_doubled[:5]]) if most_doubled else "none"
        print(f"""VOCAL DOUBLES (Mid-Side Analysis)
  Total doubled words: {doubles.get('total', 0)}
  Clean doubles: {doubles.get('clean_doubles', 0)}
  Percentage: {doubles.get('doubled_percentage', 0):.1f}%
  Beat vocals filtered: {doubles.get('beat_vocals_filtered', 0)}
  Most doubled: {words_str}
""")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <audio_file> [artist]")
        sys.exit(1)

    audio_file = sys.argv[1]
    artist = sys.argv[2] if len(sys.argv) > 2 else None

    result = analyze_track(audio_file, artist=artist)
    print_analysis_summary(result)

    # Save to JSON
    output_path = Path(audio_file).stem + "_analysis.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nFull analysis saved to: {output_path}")
