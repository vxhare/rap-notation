"""
rhyme_detector.py - Phonetic Rhyme Detection for Rap

Detects rhyme relationships between syllables using phonetic matching.

Handles:
- Perfect rhymes (cat/hat)
- Slant rhymes (cat/bed)
- Assonance (cat/man)
- Consonance (cat/kit)
- Multisyllabic rhymes (fantastic/gymnastic)
- Internal rhymes (within lines)
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from collections import defaultdict


# =============================================================================
# PHONETIC CONSTANTS
# =============================================================================

VOWELS = {
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY',
    'EH', 'ER', 'EY', 'IH', 'IY',
    'OW', 'OY', 'UH', 'UW'
}

VOWEL_GROUPS = {
    'front_high': {'IY', 'IH'},
    'front_mid': {'EY', 'EH'},
    'front_low': {'AE'},
    'central': {'AH', 'ER'},
    'back_high': {'UW', 'UH'},
    'back_mid': {'OW', 'AO'},
    'back_low': {'AA'},
    'diphthong_i': {'AY', 'OY'},
    'diphthong_u': {'AW'},
}

VOWEL_TO_GROUP = {}
for group, vowels in VOWEL_GROUPS.items():
    for v in vowels:
        VOWEL_TO_GROUP[v] = group

CONSONANT_GROUPS = {
    'stops_voiced': {'B', 'D', 'G'},
    'stops_voiceless': {'P', 'T', 'K'},
    'fricatives_voiced': {'V', 'DH', 'Z', 'ZH'},
    'fricatives_voiceless': {'F', 'TH', 'S', 'SH'},
    'nasals': {'M', 'N', 'NG'},
    'liquids': {'L', 'R'},
    'glides': {'W', 'Y'},
    'affricates': {'CH', 'JH'},
}

CONSONANT_TO_GROUP = {}
for group, consonants in CONSONANT_GROUPS.items():
    for c in consonants:
        CONSONANT_TO_GROUP[c] = group


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class RhymeType(Enum):
    PERFECT = "perfect"
    SLANT = "slant"
    ASSONANCE = "assonance"
    CONSONANCE = "consonance"
    MULTISYLLABIC = "multi"


@dataclass
class RhymeSound:
    """The rhyme-able part of a syllable."""
    syllable_id: str
    onset: list
    nucleus: str
    coda: list
    stressed: bool

    @property
    def rime(self) -> tuple:
        return (self.nucleus, tuple(self.coda))

    @property
    def rime_str(self) -> str:
        return f"{self.nucleus}_{'-'.join(self.coda)}"


@dataclass
class RhymePair:
    """A pair of syllables that rhyme"""
    syllable_id_1: str
    syllable_id_2: str
    rhyme_type: RhymeType
    score: float
    distance_beats: float


@dataclass
class RhymeGroup:
    """A group of syllables that all rhyme together"""
    id: str
    rhyme_type: RhymeType
    syllable_ids: list
    representative_sound: str
    score: float
    positions: list = field(default_factory=list)
    span_bars: int = 0
    is_end_rhyme: bool = False
    is_internal: bool = False


@dataclass
class RhymeScheme:
    """Complete rhyme analysis"""
    groups: list
    pairs: list
    total_rhymes: int = 0
    rhyme_density: float = 0.0
    multisyllabic_count: int = 0
    internal_count: int = 0
    end_rhyme_pattern: str = ""


# =============================================================================
# CORE RHYME DETECTOR
# =============================================================================

class RhymeDetector:
    """
    Detects rhymes in a list of syllables.

    Usage:
        detector = RhymeDetector()
        scheme = detector.analyze(syllables)
    """

    def __init__(
        self,
        min_score: float = 0.7,
        max_distance_bars: int = 4,
        detect_multisyllabic: bool = True,
        detect_internal: bool = True,
        genre: str = None
    ):
        # Apply genre-specific config if provided
        if genre:
            try:
                import sys
                sys.path.insert(0, '.')
                from config.genre_profiles import get_profile
                from models import Genre
                genre_enum = Genre(genre) if isinstance(genre, str) else genre
                profile = get_profile(genre_enum)
                min_score = profile.rhyme_min_score
                detect_multisyllabic = profile.expect_multisyllabic
                detect_internal = profile.expect_internal_rhymes
            except (ImportError, ValueError):
                pass

        self.min_score = min_score
        self.max_distance_bars = max_distance_bars
        self.detect_multisyllabic = detect_multisyllabic
        self.detect_internal = detect_internal
        self.genre = genre

    def analyze(
        self,
        syllables: list,
        beats_per_bar: int = 4
    ) -> RhymeScheme:
        """Analyze syllables for rhyme relationships."""

        sounds = self._extract_rhyme_sounds(syllables)
        pairs = self._find_rhyme_pairs(sounds, syllables, beats_per_bar)

        if self.detect_multisyllabic:
            multi_pairs = self._find_multisyllabic_rhymes(sounds, syllables)
            pairs.extend(multi_pairs)

        groups = self._cluster_into_groups(pairs, sounds)
        self._classify_groups(groups, syllables, beats_per_bar)
        scheme = self._build_scheme(groups, pairs, syllables, beats_per_bar)

        return scheme

    def _extract_rhyme_sounds(self, syllables: list) -> dict:
        """Extract the rhyme-relevant sounds from each syllable"""
        sounds = {}

        for syl in syllables:
            phonemes = syl.get('phonemes', [])
            if not phonemes:
                continue

            vowel_idx = None
            for i, phone in enumerate(phonemes):
                base = phone.rstrip('012')
                if base in VOWELS:
                    vowel_idx = i
                    break

            if vowel_idx is None:
                continue

            onset = phonemes[:vowel_idx]
            nucleus = phonemes[vowel_idx].rstrip('012')
            coda = phonemes[vowel_idx + 1:]
            stressed = phonemes[vowel_idx].endswith('1')

            sounds[syl['id']] = RhymeSound(
                syllable_id=syl['id'],
                onset=onset,
                nucleus=nucleus,
                coda=coda,
                stressed=stressed
            )

        return sounds

    # Common function words to skip
    SKIP_WORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'of', 'to', 'in', 'on', 'at',
        'is', 'it', 'as', 'by', 'be', 'we', 'he', 'so', 'no', 'do', 'my', 'me',
        'up', 'i', 'am', 'are', 'was', 'has', 'had', 'get', 'got', 'can', 'will',
        'just', 'like', 'that', 'this', 'with', 'for', 'you', 'your', 'they',
    }

    def _find_rhyme_pairs(self, sounds: dict, syllables: list, beats_per_bar: int) -> list:
        """Find all pairs of syllables that rhyme"""
        pairs = []
        syl_lookup = {s['id']: s for s in syllables}
        sound_ids = list(sounds.keys())

        for i, id1 in enumerate(sound_ids):
            for id2 in sound_ids[i + 1:]:
                sound1 = sounds[id1]
                sound2 = sounds[id2]
                syl1 = syl_lookup[id1]
                syl2 = syl_lookup[id2]

                # Skip identical words (same text)
                text1 = syl1.get('text', '').lower().strip('.,!?;:\'"')
                text2 = syl2.get('text', '').lower().strip('.,!?;:\'"')
                if text1 == text2:
                    continue

                # Skip very short common function words
                if text1 in self.SKIP_WORDS or text2 in self.SKIP_WORDS:
                    continue

                bar_dist = abs(syl1.get('bar', 0) - syl2.get('bar', 0))
                if bar_dist > self.max_distance_bars:
                    continue

                rhyme_type, score = self._score_rhyme(sound1, sound2)

                if score >= self.min_score:
                    beat1 = syl1.get('bar', 0) * beats_per_bar + syl1.get('beat', 0)
                    beat2 = syl2.get('bar', 0) * beats_per_bar + syl2.get('beat', 0)
                    distance = abs(beat2 - beat1)

                    pairs.append(RhymePair(
                        syllable_id_1=id1,
                        syllable_id_2=id2,
                        rhyme_type=rhyme_type,
                        score=score,
                        distance_beats=distance
                    ))

        return pairs

    def _score_rhyme(self, sound1: RhymeSound, sound2: RhymeSound) -> tuple:
        """Score how well two sounds rhyme."""

        if sound1.rime == sound2.rime:
            if sound1.onset != sound2.onset:
                return RhymeType.PERFECT, 1.0
            else:
                return RhymeType.PERFECT, 0.9

        vowel_match = sound1.nucleus == sound2.nucleus
        coda_score = self._coda_similarity(sound1.coda, sound2.coda)
        vowel_similar = self._vowels_similar(sound1.nucleus, sound2.nucleus)

        if vowel_match and coda_score < 0.8:
            return RhymeType.ASSONANCE, 0.6 + (coda_score * 0.2)

        # Consonance only for very strong coda match
        if not vowel_match and coda_score > 0.9:
            bonus = 0.1 if vowel_similar else 0
            return RhymeType.CONSONANCE, 0.6 + bonus

        # Slant rhyme needs vowel similarity + decent coda
        if vowel_similar and coda_score > 0.6:
            score = 0.5 + (coda_score * 0.25) + (0.15 if vowel_match else 0)
            return RhymeType.SLANT, score

        # No rhyme - return 0
        return RhymeType.SLANT, 0.0

    def _coda_similarity(self, coda1: list, coda2: list) -> float:
        """Score similarity between two codas"""
        if coda1 == coda2:
            return 1.0
        if not coda1 and not coda2:
            return 1.0
        if not coda1 or not coda2:
            return 0.0

        max_len = max(len(coda1), len(coda2))
        matches = 0

        for c1, c2 in zip(coda1, coda2):
            if c1 == c2:
                matches += 1
            elif self._consonants_similar(c1, c2):
                matches += 0.7

        length_penalty = abs(len(coda1) - len(coda2)) * 0.2
        return max(0, (matches / max_len) - length_penalty)

    def _vowels_similar(self, v1: str, v2: str) -> bool:
        return VOWEL_TO_GROUP.get(v1) == VOWEL_TO_GROUP.get(v2)

    def _consonants_similar(self, c1: str, c2: str) -> bool:
        return CONSONANT_TO_GROUP.get(c1) == CONSONANT_TO_GROUP.get(c2)

    def _find_multisyllabic_rhymes(self, sounds: dict, syllables: list) -> list:
        """Detect multisyllabic rhymes"""
        pairs = []
        syl_lookup = {s['id']: s for s in syllables}

        words = defaultdict(list)
        for syl in syllables:
            word_id = syl.get('word_id') or syl.get('word', syl['id'])
            words[word_id].append(syl)

        for word_id in words:
            words[word_id].sort(key=lambda s: s.get('start', 0))

        word_list = list(words.values())

        for i, word1_syls in enumerate(word_list):
            for word2_syls in word_list[i + 1:]:
                for n in range(2, 5):
                    if len(word1_syls) < n or len(word2_syls) < n:
                        continue

                    ending1 = word1_syls[-n:]
                    ending2 = word2_syls[-n:]

                    total_score = 0
                    for s1, s2 in zip(ending1, ending2):
                        id1, id2 = s1['id'], s2['id']
                        if id1 not in sounds or id2 not in sounds:
                            break
                        _, score = self._score_rhyme(sounds[id1], sounds[id2])
                        total_score += score
                    else:
                        avg_score = total_score / n
                        if avg_score >= self.min_score:
                            pairs.append(RhymePair(
                                syllable_id_1=ending1[-1]['id'],
                                syllable_id_2=ending2[-1]['id'],
                                rhyme_type=RhymeType.MULTISYLLABIC,
                                score=avg_score,
                                distance_beats=0
                            ))

        return pairs

    def _cluster_into_groups(self, pairs: list, sounds: dict) -> list:
        """Cluster rhyme pairs into groups using union-find"""
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for pair in pairs:
            union(pair.syllable_id_1, pair.syllable_id_2)

        clusters = defaultdict(set)
        for syl_id in parent:
            root = find(syl_id)
            clusters[root].add(syl_id)

        groups = []
        group_id = 0

        for root, syl_ids in clusters.items():
            if len(syl_ids) < 2:
                continue

            type_counts = defaultdict(int)
            total_score = 0
            count = 0

            for pair in pairs:
                if pair.syllable_id_1 in syl_ids and pair.syllable_id_2 in syl_ids:
                    type_counts[pair.rhyme_type] += 1
                    total_score += pair.score
                    count += 1

            dominant_type = max(type_counts, key=type_counts.get) if type_counts else RhymeType.SLANT
            avg_score = total_score / count if count > 0 else 0.5

            rep_sound = ""
            if root in sounds:
                rep_sound = sounds[root].rime_str

            groups.append(RhymeGroup(
                id=f"rhyme_{group_id}",
                rhyme_type=dominant_type,
                syllable_ids=list(syl_ids),
                representative_sound=rep_sound,
                score=avg_score
            ))
            group_id += 1

        return groups

    def _classify_groups(self, groups: list, syllables: list, beats_per_bar: int):
        """Classify each group as end-rhyme vs internal"""
        syl_lookup = {s['id']: s for s in syllables}

        for group in groups:
            positions = []
            end_count = 0

            for syl_id in group.syllable_ids:
                syl = syl_lookup.get(syl_id, {})
                bar = syl.get('bar', 0)
                beat = syl.get('beat', 0)
                positions.append((bar, beat))

                if beat >= beats_per_bar - 0.5:
                    end_count += 1

            group.positions = positions

            if positions:
                bars = [p[0] for p in positions]
                group.span_bars = max(bars) - min(bars) + 1

            total = len(group.syllable_ids)
            group.is_end_rhyme = end_count / total > 0.6
            group.is_internal = end_count / total < 0.4

    def _build_scheme(self, groups: list, pairs: list, syllables: list, beats_per_bar: int) -> RhymeScheme:
        """Build the final RhymeScheme object"""
        rhyming_syl_ids = set()
        for group in groups:
            rhyming_syl_ids.update(group.syllable_ids)

        total_rhymes = len(rhyming_syl_ids)
        bars = set(s.get('bar', 0) for s in syllables)
        num_bars = max(bars) - min(bars) + 1 if bars else 1
        rhyme_density = total_rhymes / num_bars

        multi_count = sum(1 for g in groups if g.rhyme_type == RhymeType.MULTISYLLABIC)
        internal_count = sum(1 for g in groups if g.is_internal)
        pattern = self._generate_end_rhyme_pattern(groups, syllables)

        return RhymeScheme(
            groups=groups,
            pairs=pairs,
            total_rhymes=total_rhymes,
            rhyme_density=rhyme_density,
            multisyllabic_count=multi_count,
            internal_count=internal_count,
            end_rhyme_pattern=pattern
        )

    def _generate_end_rhyme_pattern(self, groups: list, syllables: list) -> str:
        """Generate traditional rhyme scheme notation"""
        syl_lookup = {s['id']: s for s in syllables}
        bar_end_syls = defaultdict(str)

        for syl in syllables:
            bar = syl.get('bar', 0)
            beat = syl.get('beat', 0)
            if bar not in bar_end_syls or beat > syl_lookup[bar_end_syls[bar]].get('beat', 0):
                bar_end_syls[bar] = syl['id']

        syl_to_group = {}
        for group in groups:
            for syl_id in group.syllable_ids:
                syl_to_group[syl_id] = group.id

        group_to_letter = {}
        next_letter = 0
        pattern = ""

        for bar in sorted(bar_end_syls.keys()):
            syl_id = bar_end_syls[bar]
            group_id = syl_to_group.get(syl_id)

            if group_id is None:
                pattern += "x"
            else:
                if group_id not in group_to_letter:
                    group_to_letter[group_id] = chr(ord('A') + next_letter)
                    next_letter += 1
                pattern += group_to_letter[group_id]

        return pattern


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def detect_rhymes(syllables: list) -> dict:
    """Simple interface for rhyme detection."""
    detector = RhymeDetector()
    scheme = detector.analyze(syllables)

    return {
        "groups": [
            {
                "id": g.id,
                "type": g.rhyme_type.value,
                "syllables": g.syllable_ids,
                "sound": g.representative_sound,
                "score": g.score,
                "is_end_rhyme": g.is_end_rhyme,
                "is_internal": g.is_internal,
                "span_bars": g.span_bars
            }
            for g in scheme.groups
        ],
        "metrics": {
            "total_rhymes": scheme.total_rhymes,
            "density": scheme.rhyme_density,
            "multisyllabic_count": scheme.multisyllabic_count,
            "internal_count": scheme.internal_count,
            "pattern": scheme.end_rhyme_pattern
        }
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    test_syllables = [
        {"id": "s1", "phonemes": ["AY1"], "bar": 1, "beat": 1.0, "word": "I"},
        {"id": "s2", "phonemes": ["G", "AA1", "T"], "bar": 1, "beat": 1.5, "word": "got"},
        {"id": "s3", "phonemes": ["DH", "AH0"], "bar": 1, "beat": 2.0, "word": "the"},
        {"id": "s4", "phonemes": ["S", "T", "R", "AE1", "P"], "bar": 1, "beat": 2.5, "word": "strap"},
        {"id": "s5", "phonemes": ["AY1"], "bar": 1, "beat": 3.0, "word": "I"},
        {"id": "s6", "phonemes": ["G", "AA1", "T"], "bar": 1, "beat": 3.25, "word": "got"},
        {"id": "s7", "phonemes": ["DH", "AH0"], "bar": 1, "beat": 3.5, "word": "the"},
        {"id": "s8", "phonemes": ["M", "AE1", "K"], "bar": 1, "beat": 4.0, "word": "MAC"},
        {"id": "s9", "phonemes": ["AY1", "M"], "bar": 2, "beat": 1.0, "word": "I'm"},
        {"id": "s10", "phonemes": ["IH0", "N"], "bar": 2, "beat": 1.5, "word": "in"},
        {"id": "s11", "phonemes": ["DH", "AH0"], "bar": 2, "beat": 2.0, "word": "the"},
        {"id": "s12", "phonemes": ["B", "AE1", "K"], "bar": 2, "beat": 2.5, "word": "back"},
        {"id": "s13", "phonemes": ["W", "IH1", "TH"], "bar": 2, "beat": 3.0, "word": "with"},
        {"id": "s14", "phonemes": ["DH", "AH0"], "bar": 2, "beat": 3.5, "word": "the"},
        {"id": "s15", "phonemes": ["P", "AE1", "K"], "bar": 2, "beat": 4.0, "word": "pack"},
    ]

    print("Testing rhyme detection...\n")

    result = detect_rhymes(test_syllables)

    print(f"End-rhyme pattern: {result['metrics']['pattern']}")
    print(f"Rhyme density: {result['metrics']['density']:.2f} per bar")
    print(f"Total rhyming syllables: {result['metrics']['total_rhymes']}")
    print()

    print("Rhyme groups found:")
    for group in result['groups']:
        syls = ", ".join(group['syllables'])
        print(f"  [{group['type']}] {group['sound']}: {syls}")
