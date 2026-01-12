"""
Validate rhyme detection against MCFlow ground truth.
"""
import sys
sys.path.insert(0, '.')

from collections import defaultdict
from parsers.mcflow_parser import MCFlowParser
from phonetics.phoneme_lookup import get_phonemes, get_rime
from analysis.rhyme_detector import RhymeDetector


def validate_rhyme_detection(rap_file: str):
    """Compare our detection vs MCFlow ground truth."""

    # Parse file
    parser = MCFlowParser()
    verse = parser.parse_file(rap_file)

    print(f"\n{'='*60}")
    print(f"RHYME VALIDATION: {verse.artist} - {verse.title}")
    print(f"{'='*60}\n")

    # Extract ground truth rhyme groups
    gt_groups = defaultdict(list)
    gt_syllables = {}

    for syl in verse.syllables:
        gt_syllables[syl.id] = syl
        if syl.rhyme_class and syl.rhyme_class not in ('.', ''):
            gt_groups[syl.rhyme_class].append(syl.id)

    # Only keep groups with 2+ syllables
    gt_groups = {k: v for k, v in gt_groups.items() if len(v) >= 2}

    print(f"Ground Truth:")
    print(f"  Rhyme groups: {len(gt_groups)}")
    print(f"  Rhyming syllables: {sum(len(v) for v in gt_groups.values())}")

    # Build syllable data for our detector
    syllables = []
    for syl in verse.syllables:
        phonemes = get_phonemes(syl.text)
        primary = phonemes[0] if phonemes else []
        nucleus, coda = get_rime(primary)

        syllables.append({
            'id': syl.id,
            'text': syl.text,
            'bar': syl.bar,
            'beat': syl.beat,
            'phonemes': primary,
            'nucleus': nucleus,
            'coda': coda
        })

    # Run our detector
    detector = RhymeDetector(min_score=0.7)
    scheme = detector.analyze(syllables)

    print(f"\nOur Detection:")
    print(f"  Rhyme groups: {len(scheme.groups)}")
    print(f"  Rhyming syllables: {sum(len(g.syllable_ids) for g in scheme.groups)}")

    # Compute precision/recall
    # For each GT group, check if we found a corresponding group

    gt_pairs = set()
    for group_syls in gt_groups.values():
        for i, s1 in enumerate(group_syls):
            for s2 in group_syls[i+1:]:
                pair = tuple(sorted([s1, s2]))
                gt_pairs.add(pair)

    our_pairs = set()
    for group in scheme.groups:
        syls = group.syllable_ids
        for i, s1 in enumerate(syls):
            for s2 in syls[i+1:]:
                pair = tuple(sorted([s1, s2]))
                our_pairs.add(pair)

    true_pos = len(gt_pairs & our_pairs)
    false_pos = len(our_pairs - gt_pairs)
    false_neg = len(gt_pairs - our_pairs)

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nPair-wise Accuracy:")
    print(f"  Ground truth pairs: {len(gt_pairs)}")
    print(f"  Our detected pairs: {len(our_pairs)}")
    print(f"  True positives: {true_pos}")
    print(f"  False positives: {false_pos}")
    print(f"  False negatives: {false_neg}")
    print(f"\n  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")

    # Show some examples
    print(f"\n{'='*60}")
    print("SAMPLE COMPARISONS")
    print(f"{'='*60}")

    # Show matched rhymes
    print("\n[MATCHED] GT rhymes we correctly detected:")
    matched_count = 0
    for gt_label, gt_syls in list(gt_groups.items())[:5]:
        texts = [gt_syllables[s].text for s in gt_syls[:5]]
        # Check if any of our groups contain these
        for group in scheme.groups:
            overlap = set(gt_syls) & set(group.syllable_ids)
            if len(overlap) >= 2:
                print(f"  GT[{gt_label}]: {texts} -> Found in our group")
                matched_count += 1
                break

    # Show missed rhymes
    print("\n[MISSED] GT rhymes we didn't detect:")
    missed_count = 0
    for gt_label, gt_syls in gt_groups.items():
        texts = [gt_syllables[s].text for s in gt_syls[:5]]
        found = False
        for group in scheme.groups:
            overlap = set(gt_syls) & set(group.syllable_ids)
            if len(overlap) >= 2:
                found = True
                break
        if not found:
            print(f"  GT[{gt_label}]: {texts}")
            missed_count += 1
            if missed_count >= 5:
                break

    # Show false positives (our detections not in GT)
    print("\n[EXTRA] Rhymes we detected not in GT:")
    extra_count = 0
    for group in scheme.groups[:10]:
        texts = [gt_syllables.get(s, type('obj', (), {'text': s})()).text for s in group.syllable_ids[:5]]
        # Check if this group overlaps significantly with any GT group
        max_overlap = 0
        for gt_syls in gt_groups.values():
            overlap = len(set(group.syllable_ids) & set(gt_syls))
            max_overlap = max(max_overlap, overlap)

        if max_overlap < 2:
            print(f"  [{group.rhyme_type.value}] {texts}")
            extra_count += 1
            if extra_count >= 5:
                break

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'gt_groups': len(gt_groups),
        'our_groups': len(scheme.groups)
    }


if __name__ == "__main__":
    import glob

    # Validate on a few songs
    files = glob.glob('data/MCFlow/Humdrum/Eminem_*.rap')[:3]

    results = []
    for f in files:
        try:
            result = validate_rhyme_detection(f)
            results.append(result)
        except Exception as e:
            print(f"Error on {f}: {e}")

    if results:
        print(f"\n{'='*60}")
        print("AGGREGATE RESULTS")
        print(f"{'='*60}")
        avg_p = sum(r['precision'] for r in results) / len(results)
        avg_r = sum(r['recall'] for r in results) / len(results)
        avg_f1 = sum(r['f1'] for r in results) / len(results)
        print(f"Average Precision: {avg_p:.3f}")
        print(f"Average Recall: {avg_r:.3f}")
        print(f"Average F1: {avg_f1:.3f}")
