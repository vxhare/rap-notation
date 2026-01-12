"""
ground_truth.py - Unified ground truth data structures and loader

Loads and unifies data from MCFlow and flowBook datasets.
"""

from dataclasses import dataclass, field
from typing import Optional, Iterator
from pathlib import Path


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GroundTruthSyllable:
    """Unified syllable format from either dataset"""
    id: str
    text: str
    word: str

    bar: int
    beat: float
    duration_beats: float
    quantization: int

    phonemes: list = field(default_factory=list)
    stressed: bool = False
    break_level: int = 0
    line_ending: bool = False
    breath_ending: bool = False

    rhyme_class: Optional[str] = None
    rhyme_index: Optional[int] = None

    source: str = ""


@dataclass
class GroundTruthVerse:
    """A complete verse with metadata"""
    id: str
    artist: str
    title: str
    syllables: list

    bpm: Optional[float] = None
    time_signature: tuple = (4, 4)

    source: str = ""
    source_file: str = ""

    @property
    def total_bars(self) -> int:
        if not self.syllables:
            return 0
        return max(s.bar for s in self.syllables)

    @property
    def rhyme_classes(self) -> set:
        return {s.rhyme_class for s in self.syllables if s.rhyme_class}


# =============================================================================
# UNIFIED LOADER
# =============================================================================

class GroundTruthLoader:
    """
    Unified loader for all ground truth datasets.

    Usage:
        loader = GroundTruthLoader(
            mcflow_dir="path/to/MCFlow/Humdrum",
            flowbook_dir="path/to/flowBook/SourceData/VerseTranscriptions"
        )

        for verse in loader.load_all():
            print(f"{verse.artist} - {verse.title}")
    """

    def __init__(
        self,
        mcflow_dir: Optional[str] = None,
        flowbook_dir: Optional[str] = None
    ):
        self.mcflow_dir = mcflow_dir
        self.flowbook_dir = flowbook_dir
        self._verses: dict = {}
        self._mcflow = None
        self._flowbook = None

    def _get_mcflow(self):
        if self._mcflow is None and self.mcflow_dir:
            from .mcflow_parser import MCFlowParser
            self._mcflow = MCFlowParser(self.mcflow_dir)
        return self._mcflow

    def _get_flowbook(self):
        if self._flowbook is None and self.flowbook_dir:
            from .flowbook_parser import FlowBookParser
            self._flowbook = FlowBookParser(self.flowbook_dir)
        return self._flowbook

    def load_all(self) -> Iterator[GroundTruthVerse]:
        """Load all verses from all datasets"""
        mcflow = self._get_mcflow()
        if mcflow:
            for verse in mcflow.parse_all():
                self._verses[verse.id] = verse
                yield verse

        flowbook = self._get_flowbook()
        if flowbook:
            for verse in flowbook.parse_all():
                self._verses[verse.id] = verse
                yield verse

    def get_verse(self, verse_id: str) -> Optional[GroundTruthVerse]:
        """Get a specific verse by ID"""
        if verse_id not in self._verses:
            list(self.load_all())
        return self._verses.get(verse_id)

    def get_by_artist(self, artist: str) -> list:
        """Get all verses by an artist"""
        if not self._verses:
            list(self.load_all())

        artist_lower = artist.lower()
        return [
            v for v in self._verses.values()
            if artist_lower in v.artist.lower()
        ]

    def get_rhyme_data(self, verse_id: str) -> dict:
        """Extract rhyme ground truth for validation."""
        verse = self.get_verse(verse_id)
        if not verse:
            return {}

        syllable_rhymes = {}
        rhyme_groups = {}

        for syl in verse.syllables:
            if syl.rhyme_class:
                syllable_rhymes[syl.id] = syl.rhyme_class

                if syl.rhyme_class not in rhyme_groups:
                    rhyme_groups[syl.rhyme_class] = []
                rhyme_groups[syl.rhyme_class].append(syl.id)

        total = len(verse.syllables)
        rhyming = len(syllable_rhymes)
        density = rhyming / total if total > 0 else 0

        return {
            "syllable_rhymes": syllable_rhymes,
            "rhyme_groups": rhyme_groups,
            "rhyme_density": density,
            "num_groups": len(rhyme_groups),
            "total_syllables": total,
            "rhyming_syllables": rhyming
        }

    def get_timing_data(self, verse_id: str) -> list:
        """Extract timing ground truth for validation."""
        verse = self.get_verse(verse_id)
        if not verse:
            return []

        return [
            {
                "id": syl.id,
                "text": syl.text,
                "bar": syl.bar,
                "beat": syl.beat,
                "duration": syl.duration_beats,
                "quantization": syl.quantization
            }
            for syl in verse.syllables
        ]


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def compare_rhyme_detection(ground_truth: dict, detected: dict) -> dict:
    """Compare detected rhymes against ground truth."""
    gt_pairs = set()
    for group_syls in ground_truth.get("rhyme_groups", {}).values():
        for i, s1 in enumerate(group_syls):
            for s2 in group_syls[i+1:]:
                gt_pairs.add((min(s1, s2), max(s1, s2)))

    detected_pairs = set()
    for group in detected.get("groups", []):
        syls = group.get("syllables", [])
        for i, s1 in enumerate(syls):
            for s2 in syls[i+1:]:
                detected_pairs.add((min(s1, s2), max(s1, s2)))

    true_positives = len(gt_pairs & detected_pairs)
    false_positives = len(detected_pairs - gt_pairs)
    false_negatives = len(gt_pairs - detected_pairs)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def dataset_statistics(loader: GroundTruthLoader) -> dict:
    """Generate statistics about the combined dataset"""
    verses = list(loader.load_all())

    total_syllables = sum(len(v.syllables) for v in verses)
    total_bars = sum(v.total_bars for v in verses)
    artists = set(v.artist for v in verses)

    rhyme_densities = []
    for v in verses:
        rhyme_data = loader.get_rhyme_data(v.id)
        if rhyme_data:
            rhyme_densities.append(rhyme_data["rhyme_density"])

    avg_rhyme_density = sum(rhyme_densities) / len(rhyme_densities) if rhyme_densities else 0

    mcflow_count = sum(1 for v in verses if v.source == "mcflow")
    flowbook_count = sum(1 for v in verses if v.source == "flowbook")

    return {
        "total_verses": len(verses),
        "total_syllables": total_syllables,
        "total_bars": total_bars,
        "unique_artists": len(artists),
        "avg_rhyme_density": avg_rhyme_density,
        "mcflow_verses": mcflow_count,
        "flowbook_verses": flowbook_count,
        "avg_syllables_per_verse": total_syllables / len(verses) if verses else 0,
        "avg_syllables_per_bar": total_syllables / total_bars if total_bars else 0
    }
