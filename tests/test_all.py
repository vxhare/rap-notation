"""
test_all.py - Comprehensive tests for rap notation system

Run with: python -m pytest tests/test_all.py -v
Or just: python tests/test_all.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_phoneme_lookup():
    """Test phoneme lookup with CMU dict and fallbacks"""
    print("\n" + "="*60)
    print("TEST: Phoneme Lookup")
    print("="*60)

    from phonetics.phoneme_lookup import get_phonemes, get_rime, phonemize_lyrics

    # Standard words
    cat_phones = get_phonemes("cat")
    assert cat_phones, "Should find 'cat' in CMU dict"
    assert cat_phones[0] == ["K", "AE1", "T"], f"Expected K AE1 T, got {cat_phones[0]}"
    print(f"  cat: {' '.join(cat_phones[0])}")

    # Custom slang
    skrrt_phones = get_phonemes("skrrt")
    assert skrrt_phones, "Should find 'skrrt' in custom dict"
    print(f"  skrrt: {' '.join(skrrt_phones[0])}")

    # Rime extraction
    nucleus, coda = get_rime(["K", "AE1", "T"])
    assert nucleus == "AE", f"Expected AE, got {nucleus}"
    assert coda == ["T"], f"Expected ['T'], got {coda}"
    print(f"  rime of 'cat': nucleus={nucleus}, coda={coda}")

    # Batch processing
    words = ["cat", "hat", "running", "skrrt"]
    results = phonemize_lyrics(words)
    assert len(results) == 4
    print(f"  Batch processed {len(results)} words")

    print("  PASSED")
    return True


def test_rhyme_detection():
    """Test rhyme detection algorithm"""
    print("\n" + "="*60)
    print("TEST: Rhyme Detection")
    print("="*60)

    from analysis.rhyme_detector import detect_rhymes, RhymeDetector

    # Test syllables with known rhymes
    test_syllables = [
        {"id": "s1", "phonemes": ["K", "AE1", "T"], "bar": 1, "beat": 2.0, "text": "cat"},
        {"id": "s2", "phonemes": ["HH", "AE1", "T"], "bar": 1, "beat": 4.0, "text": "hat"},
        {"id": "s3", "phonemes": ["B", "AE1", "T"], "bar": 2, "beat": 2.0, "text": "bat"},
        {"id": "s4", "phonemes": ["R", "AH1", "N"], "bar": 2, "beat": 4.0, "text": "run"},
        {"id": "s5", "phonemes": ["S", "AH1", "N"], "bar": 3, "beat": 2.0, "text": "sun"},
        {"id": "s6", "phonemes": ["F", "AH1", "N"], "bar": 3, "beat": 4.0, "text": "fun"},
    ]

    result = detect_rhymes(test_syllables)

    assert len(result['groups']) >= 2, "Should find at least 2 rhyme groups"
    print(f"  Found {len(result['groups'])} rhyme groups")

    # Check that cat/hat/bat are grouped
    at_group = None
    for g in result['groups']:
        if 's1' in g['syllables'] and 's2' in g['syllables']:
            at_group = g
            break

    assert at_group is not None, "cat/hat should be in same group"
    print(f"  'cat/hat/bat' group: {at_group['syllables']}")

    print(f"  Rhyme pattern: {result['metrics']['pattern']}")
    print(f"  Rhyme density: {result['metrics']['density']:.2f}")

    print("  PASSED")
    return True


def test_grid_alignment():
    """Test grid alignment and microtiming"""
    print("\n" + "="*60)
    print("TEST: Grid Alignment")
    print("="*60)

    from analysis.grid_alignment import align_and_analyze, GridAligner, GridType

    # Syllables at 90 BPM
    bpm = 90
    beat_duration = 60 / bpm  # ~0.667 seconds

    test_syllables = [
        {"id": "s1", "text": "I", "start": 0.0},
        {"id": "s2", "text": "got", "start": beat_duration * 0.25},  # On 16th note
        {"id": "s3", "text": "the", "start": beat_duration * 0.5},
        {"id": "s4", "text": "strap", "start": beat_duration * 0.75},
        {"id": "s5", "text": "I", "start": beat_duration * 1.0},
        {"id": "s6", "text": "got", "start": beat_duration * 1.25},
        {"id": "s7", "text": "the", "start": beat_duration * 1.5},
        {"id": "s8", "text": "MAC", "start": beat_duration * 1.75},
    ]

    result = align_and_analyze(test_syllables, bpm)

    assert 'syllables' in result
    assert len(result['syllables']) == 8
    print(f"  Aligned {len(result['syllables'])} syllables")

    # Check grid comparison
    assert 'grid_comparison' in result
    print(f"  Recommended grid: {result['grid_comparison']['recommended']}")
    print(f"  16th fit score: {result['grid_comparison']['sixteenth']['fit_score']:.3f}")
    print(f"  Noctuplet fit score: {result['grid_comparison']['noctuplet']['fit_score']:.3f}")

    # Check microtiming
    assert 'microtiming' in result
    print(f"  Microtiming style: {result['microtiming']['style']}")
    print(f"  Mean deviation: {result['microtiming']['mean_deviation_ms']:.1f}ms")

    print("  PASSED")
    return True


def test_flow_metrics():
    """Test flow analysis metrics"""
    print("\n" + "="*60)
    print("TEST: Flow Metrics")
    print("="*60)

    from analysis.flow_metrics import analyze_flow, RhymeEntropyAnalyzer

    # Test syllables with rhymes
    test_syllables = []

    # Bar 1
    for i, (beat, text, rhyme) in enumerate([
        (1.0, "I", None), (1.5, "got", None), (2.0, "the", None),
        (2.5, "strap", "A"), (3.0, "I", None), (3.5, "got", None),
        (4.0, "MAC", "A")
    ]):
        test_syllables.append({
            "id": f"s{i}", "text": text, "bar": 1, "beat": beat,
            "grid_slot": int((beat - 1) * 4), "rhyme_class": rhyme,
            "stressed": text in ["strap", "MAC"]
        })

    # Bar 2
    for i, (beat, text, rhyme) in enumerate([
        (1.0, "in", None), (1.5, "the", None), (2.0, "back", "A"),
        (2.5, "with", None), (3.0, "the", None), (3.5, "pack", "A")
    ]):
        test_syllables.append({
            "id": f"s{len(test_syllables)}", "text": text, "bar": 2, "beat": beat,
            "grid_slot": int((beat - 1) * 4), "rhyme_class": rhyme,
            "stressed": text in ["back", "pack"]
        })

    result = analyze_flow(test_syllables)

    # Check density
    assert 'density' in result
    assert result['density']['syllables_per_bar'] > 0
    print(f"  Syllables/bar: {result['density']['syllables_per_bar']:.1f}")

    # Check rhyme metrics
    assert 'rhyme' in result
    print(f"  Rhyme density: {result['rhyme']['density']:.1f}")
    print(f"  Rhyme entropy: {result['rhyme']['entropy']:.3f}")
    print(f"  Pattern type: {result['rhyme']['pattern_type']}")

    # Check rhythm
    assert 'rhythm' in result
    print(f"  Groove complexity: {result['rhythm']['complexity']:.3f}")
    print(f"  Syncopation: {result['rhythm']['syncopation']:.1%}")
    print(f"  Style: {result['rhythm']['style']}")

    print("  PASSED")
    return True


def test_parser_structures():
    """Test parser data structures (without actual files)"""
    print("\n" + "="*60)
    print("TEST: Parser Structures")
    print("="*60)

    from parsers.ground_truth import GroundTruthSyllable, GroundTruthVerse

    # Create test syllables
    syllables = [
        GroundTruthSyllable(
            id="syl_0", text="test", word="test",
            bar=1, beat=1.0, duration_beats=0.25, quantization=16,
            rhyme_class="A", source="test"
        ),
        GroundTruthSyllable(
            id="syl_1", text="best", word="best",
            bar=1, beat=2.0, duration_beats=0.25, quantization=16,
            rhyme_class="A", source="test"
        ),
    ]

    # Create test verse
    verse = GroundTruthVerse(
        id="test_verse",
        artist="Test Artist",
        title="Test Song",
        syllables=syllables,
        bpm=90.0,
        source="test"
    )

    assert verse.total_bars == 1
    assert verse.rhyme_classes == {"A"}
    print(f"  Created verse: {verse.artist} - {verse.title}")
    print(f"  Total bars: {verse.total_bars}")
    print(f"  Rhyme classes: {verse.rhyme_classes}")

    print("  PASSED")
    return True


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "#"*60)
    print("# RAP NOTATION SYSTEM - TEST SUITE")
    print("#"*60)

    tests = [
        ("Phoneme Lookup", test_phoneme_lookup),
        ("Rhyme Detection", test_rhyme_detection),
        ("Grid Alignment", test_grid_alignment),
        ("Flow Metrics", test_flow_metrics),
        ("Parser Structures", test_parser_structures),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  FAILED: {e}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, p, _ in results if p)
    total = len(results)

    for name, p, err in results:
        status = "PASS" if p else "FAIL"
        print(f"  [{status}] {name}")
        if err:
            print(f"         Error: {err}")

    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n  {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
