"""
test_integration.py - End-to-end integration test

Tests the complete flow:
1. Parse ground truth from MCFlow
2. Add phonemes
3. Run rhyme detection
4. Run grid alignment
5. Analyze flow metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_integration_test():
    """Full pipeline integration test"""
    print("\n" + "#"*60)
    print("# INTEGRATION TEST: FULL PIPELINE")
    print("#"*60)

    # 1. Parse ground truth
    print("\n[1] Parsing ground truth from MCFlow...")
    from parsers.mcflow_parser import MCFlowParser

    parser = MCFlowParser()
    filepath = Path("data/MCFlow/Humdrum/Eminem_LoseYourself.rap")

    if not filepath.exists():
        filepath = Path("data/MCFlow/Humdrum/DrDre_NuthinButAGThang.rap")

    verse = parser.parse_file(filepath)
    print(f"    Parsed: {verse.artist} - {verse.title}")
    print(f"    Syllables: {len(verse.syllables)}")
    print(f"    Bars: {verse.total_bars}")

    # Filter out rest markers
    syllables = [s for s in verse.syllables if s.text and s.text not in ['R', '.']]
    print(f"    Lyric syllables: {len(syllables)}")

    # 2. Add phonemes
    print("\n[2] Adding phonemes...")
    from phonetics.phoneme_lookup import get_phonemes

    enriched = []
    phoneme_count = 0

    for syl in syllables:
        phones = get_phonemes(syl.text)
        syl_dict = {
            "id": syl.id,
            "text": syl.text,
            "bar": syl.bar,
            "beat": syl.beat,
            "rhyme_class": syl.rhyme_class,
            "stressed": syl.stressed,
            "phonemes": phones[0] if phones else [],
            "grid_slot": int((syl.beat - 1) * 4) if syl.beat else 0
        }
        enriched.append(syl_dict)
        if phones:
            phoneme_count += 1

    print(f"    Phonemes found for {phoneme_count}/{len(syllables)} syllables ({100*phoneme_count/len(syllables):.1f}%)")

    # 3. Run rhyme detection
    print("\n[3] Running rhyme detection...")
    from analysis.rhyme_detector import detect_rhymes

    # Only use syllables with phonemes
    with_phonemes = [s for s in enriched if s['phonemes']]
    rhyme_result = detect_rhymes(with_phonemes)

    print(f"    Rhyme groups found: {len(rhyme_result['groups'])}")
    print(f"    Rhyming syllables: {rhyme_result['metrics']['total_rhymes']}")
    print(f"    Rhyme density: {rhyme_result['metrics']['density']:.2f} per bar")
    print(f"    Pattern: {rhyme_result['metrics']['pattern'][:20]}...")

    # Show top rhyme groups
    print("\n    Top rhyme groups:")
    for group in sorted(rhyme_result['groups'], key=lambda g: -len(g['syllables']))[:3]:
        syl_texts = [s['text'] for s in enriched if s['id'] in group['syllables']][:5]
        print(f"      [{group['type']}] {', '.join(syl_texts)}")

    # 4. Run grid alignment
    print("\n[4] Running grid alignment...")
    from analysis.grid_alignment import GridAligner, GridType

    # Create synthetic timing (since we don't have audio)
    bpm = verse.bpm or 90
    beat_duration = 60 / bpm

    for syl in enriched:
        # Convert bar/beat to time
        abs_beat = (syl['bar'] - 1) * 4 + (syl['beat'] - 1)
        syl['start'] = abs_beat * beat_duration
        syl['end'] = syl['start'] + beat_duration * 0.2

    aligner = GridAligner(bpm)
    aligned = aligner.align_syllables(enriched)

    grid_comparison = aligner.compare_grids(aligned)
    microtiming = aligner.analyze_microtiming(aligned)

    print(f"    Recommended grid: {grid_comparison['recommended']}")
    print(f"    16th note fit: {grid_comparison['sixteenth']['fit_score']:.3f}")
    print(f"    Noctuplet fit: {grid_comparison['noctuplet']['fit_score']:.3f}")
    print(f"    Microtiming style: {microtiming.style}")
    print(f"    Mean deviation: {microtiming.mean_deviation_ms:.1f}ms")

    # 5. Analyze flow metrics
    print("\n[5] Analyzing flow metrics...")
    from analysis.flow_metrics import analyze_flow

    flow_result = analyze_flow(enriched)

    print(f"    Syllables/bar: {flow_result['density']['syllables_per_bar']:.1f}")
    print(f"    Syllables/beat: {flow_result['density']['syllables_per_beat']:.2f}")
    print(f"    Rhyme entropy: {flow_result['rhyme']['entropy']:.3f}")
    print(f"    Pattern type: {flow_result['rhyme']['pattern_type']}")
    print(f"    Groove complexity: {flow_result['rhythm']['complexity']:.3f}")
    print(f"    Syncopation: {flow_result['rhythm']['syncopation']:.1%}")
    print(f"    Style: {flow_result['rhythm']['style']}")

    # Summary
    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETE")
    print("="*60)
    print(f"""
    Song: {verse.artist} - {verse.title}
    BPM: {bpm}

    PARSING
      Raw syllables: {len(verse.syllables)}
      Lyric syllables: {len(syllables)}
      Phoneme coverage: {100*phoneme_count/len(syllables):.1f}%

    RHYME ANALYSIS
      Groups detected: {len(rhyme_result['groups'])}
      Rhyme density: {rhyme_result['metrics']['density']:.2f}/bar

    RHYTHM ANALYSIS
      Grid: {grid_comparison['recommended']}
      Complexity: {flow_result['rhythm']['complexity']:.3f}
      Syncopation: {flow_result['rhythm']['syncopation']:.1%}

    FLOW ANALYSIS
      Density: {flow_result['density']['syllables_per_bar']:.1f} syl/bar
      Entropy: {flow_result['rhyme']['entropy']:.3f}
      Style: {flow_result['rhythm']['style']}
    """)

    return True


if __name__ == "__main__":
    try:
        result = run_integration_test()
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
