"""
test_mcflow_real.py - Test with real MCFlow data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.mcflow_parser import MCFlowParser


def test_single_file():
    """Test parsing a single MCFlow file"""
    print("\n" + "="*60)
    print("TEST: Parse Real MCFlow File")
    print("="*60)

    parser = MCFlowParser()

    # Parse a well-known track
    filepath = Path("data/MCFlow/Humdrum/DrDre_NuthinButAGThang.rap")

    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return False

    verse = parser.parse_file(filepath)

    print(f"  Artist: {verse.artist}")
    print(f"  Title: {verse.title}")
    print(f"  BPM: {verse.bpm}")
    print(f"  Total syllables: {len(verse.syllables)}")
    print(f"  Total bars: {verse.total_bars}")
    print(f"  Rhyme classes found: {verse.rhyme_classes}")

    # Show first 10 syllables
    print("\n  First 10 syllables:")
    print(f"  {'TEXT':<12} {'BAR':<5} {'BEAT':<6} {'RHYME':<6} {'STRESS'}")
    print("  " + "-"*45)

    for syl in verse.syllables[:10]:
        rhyme = syl.rhyme_class or "-"
        stress = "*" if syl.stressed else ""
        print(f"  {syl.text:<12} {syl.bar:<5} {syl.beat:<6.2f} {rhyme:<6} {stress}")

    # Count actual lyrics (skip rests marked as 'R' or '.')
    actual_lyrics = [s for s in verse.syllables if s.text and s.text not in ['R', '.']]
    print(f"\n  Actual lyric syllables: {len(actual_lyrics)}")

    return True


def test_multiple_files():
    """Test parsing multiple MCFlow files"""
    print("\n" + "="*60)
    print("TEST: Parse Multiple MCFlow Files")
    print("="*60)

    mcflow_dir = Path("data/MCFlow/Humdrum")

    if not mcflow_dir.exists():
        print(f"  Directory not found: {mcflow_dir}")
        return False

    parser = MCFlowParser(str(mcflow_dir))

    total_verses = 0
    total_syllables = 0
    artists = set()
    failed = []

    for verse in parser.parse_all():
        total_verses += 1
        total_syllables += len(verse.syllables)
        artists.add(verse.artist)

    print(f"  Total verses parsed: {total_verses}")
    print(f"  Total syllables: {total_syllables}")
    print(f"  Unique artists: {len(artists)}")
    print(f"  Avg syllables/verse: {total_syllables/total_verses:.1f}")

    # Show some artists
    print(f"\n  Sample artists: {', '.join(list(artists)[:5])}")

    return True


def test_rhyme_extraction():
    """Test rhyme data extraction from MCFlow"""
    print("\n" + "="*60)
    print("TEST: Extract Rhyme Data")
    print("="*60)

    from parsers.ground_truth import GroundTruthLoader

    loader = GroundTruthLoader(mcflow_dir="data/MCFlow/Humdrum")

    # Load just one verse
    parser = MCFlowParser("data/MCFlow/Humdrum")
    filepath = Path("data/MCFlow/Humdrum/NotoriousBIG_Juicy.rap")

    if not filepath.exists():
        print("  File not found, trying another...")
        filepath = Path("data/MCFlow/Humdrum/Eminem_LoseYourself.rap")

    if not filepath.exists():
        print("  No suitable file found")
        return False

    verse = parser.parse_file(filepath)

    # Extract rhyme data
    rhyme_groups = {}
    for syl in verse.syllables:
        if syl.rhyme_class and syl.text not in ['R', '.']:
            if syl.rhyme_class not in rhyme_groups:
                rhyme_groups[syl.rhyme_class] = []
            rhyme_groups[syl.rhyme_class].append(syl)

    print(f"  Song: {verse.artist} - {verse.title}")
    print(f"  Rhyme classes found: {len(rhyme_groups)}")

    # Show rhyme groups
    for rhyme_class, syls in sorted(rhyme_groups.items())[:5]:
        texts = [s.text for s in syls[:6]]
        print(f"    {rhyme_class}: {', '.join(texts)}")

    return True


if __name__ == "__main__":
    tests = [
        ("Single File", test_single_file),
        ("Multiple Files", test_multiple_files),
        ("Rhyme Extraction", test_rhyme_extraction),
    ]

    for name, test_func in tests:
        try:
            result = test_func()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = "FAIL"
            print(f"  Error: {e}")

        print(f"\n  [{status}] {name}")
