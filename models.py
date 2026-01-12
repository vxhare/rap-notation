"""
Rap Notation System - Flexible Data Models

Design principles (per user guidance):
- Support multiple grid types (16th, noctuplet, raw ms)
- Track phrases SEPARATELY from bars (for enjambment, cross-rhythm)
- Handle singing vs rapping (melodicity spectrum)
- Store CONFIDENCE SCORES, not just binary decisions
- Be genre-aware (ICP ≠ Kendrick ≠ Three 6 Mafia)

NOT like MCFlow:
- No "1 line = 1 bar" assumption
- Preserve raw microtiming (don't quantize it away)
- Support modern styles (drill, SoundCloud, melodic rap)
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class GridType(Enum):
    """Rhythmic grid types"""
    RAW_MS = "raw"           # No quantization, raw milliseconds
    SIXTEENTH = "16"         # Standard 16th note grid
    NOCTUPLET = "noct"       # Connor's 9-per-half-bar
    THIRTYSECOND = "32"      # High resolution
    TRIPLET = "triplet"      # Triplet feel


class RhymeType(Enum):
    PERFECT = "perfect"
    SLANT = "slant"
    ASSONANCE = "assonance"
    CONSONANCE = "consonance"
    MULTISYLLABIC = "multi"
    INTERNAL = "internal"


class VocalStyle(Enum):
    """Spectrum from pure rap to pure singing"""
    RAP_PERCUSSIVE = "rap_percussive"    # Aggressive, rhythmic (DMX)
    RAP_MELODIC = "rap_melodic"          # Melodic rap (Drake verse)
    SING_RAP = "sing_rap"                # In between (Travis Scott)
    SINGING = "singing"                   # Full singing (hook)


class Genre(Enum):
    """Genre affects expected patterns"""
    CLASSIC_EAST = "classic_east"         # 90s NYC boom bap
    CLASSIC_WEST = "classic_west"         # G-funk
    CLASSIC_SOUTH = "classic_south"       # Early crunk, Three 6
    TRAP = "trap"                         # Modern Atlanta
    DRILL = "drill"                       # UK/Chicago drill
    SOUNDCLOUD = "soundcloud"             # Lil Peep, XXX style
    LYRICAL = "lyrical"                   # Kendrick, Cole
    HORRORCORE = "horrorcore"             # ICP, etc
    MELODIC = "melodic"                   # Juice WRLD style


# =============================================================================
# TIME - PRESERVE RAW + PROVIDE MULTIPLE GRIDS
# =============================================================================

@dataclass
class TimePoint:
    """
    A point in time with BOTH raw and quantized representations.
    Never throw away the raw data.
    """
    # Raw timing (ALWAYS populated)
    seconds: float                        # Absolute time in audio

    # Musical context (populated after beat detection)
    bar: Optional[int] = None             # Bar number (1-indexed)
    beat: Optional[float] = None          # Beat within bar (1.0 - 4.999)

    # Multiple grid positions (populated on demand)
    grid_16: Optional[int] = None         # Position in 16th grid (0-15)
    grid_noct: Optional[int] = None       # Position in noctuplet grid (0-17)
    grid_32: Optional[int] = None         # Position in 32nd grid (0-31)

    # Microtiming - THE CRITICAL DATA
    deviation_ms: Optional[float] = None  # Ms off the quantized grid
    deviation_16: Optional[float] = None  # Deviation from 16th grid
    deviation_noct: Optional[float] = None # Deviation from noctuplet grid


@dataclass
class TimeSpan:
    """A duration with start/end"""
    start: TimePoint
    end: TimePoint

    @property
    def duration_seconds(self) -> float:
        return self.end.seconds - self.start.seconds

    @property
    def duration_beats(self) -> Optional[float]:
        if self.start.beat is not None and self.end.beat is not None:
            bar_diff = (self.end.bar or 0) - (self.start.bar or 0)
            return bar_diff * 4 + (self.end.beat - self.start.beat)
        return None


# =============================================================================
# SYLLABLE - THE ATOMIC UNIT (with confidence scores)
# =============================================================================

@dataclass
class Syllable:
    """
    The atomic unit. Everything links back to syllables.
    Includes CONFIDENCE SCORES for uncertain detections.
    """
    # Identity
    id: str
    text: str
    word_id: Optional[str] = None

    # Timing - raw AND musical
    start_sec: float = 0.0
    end_sec: float = 0.0
    time: Optional[TimeSpan] = None

    # Phonetics (with confidence)
    phonemes: list = field(default_factory=list)
    phoneme_confidence: float = 1.0       # 1.0 = CMU dict, lower = G2P
    vowel: Optional[str] = None
    stressed: bool = False
    stress_confidence: float = 1.0

    # Pitch (for melodicity detection)
    pitch_hz: Optional[float] = None
    pitch_confidence: float = 0.0         # 0 = no pitch detected
    pitch_variance: float = 0.0           # High = singing, low = rap
    pitch_contour: list = field(default_factory=list)

    # Vocal style
    vocal_style: VocalStyle = VocalStyle.RAP_PERCUSSIVE
    melodicity_score: float = 0.0         # 0 = pure rap, 1 = pure singing

    # Microtiming
    velocity: float = 1.0
    quantized_position: Optional[int] = None
    deviation_ms: float = 0.0

    # Rhyme participation (populated by rhyme detector)
    rhyme_group_ids: list = field(default_factory=list)
    is_rhyme_anchor: bool = False         # Is this a "target" rhyme position?


# =============================================================================
# PHRASE - INDEPENDENT FROM BARS (for enjambment)
# =============================================================================

@dataclass
class Phrase:
    """
    A grammatical/breath unit - NOT tied to bar boundaries.
    Allows enjambment (phrase crossing bar lines).
    """
    id: str
    syllables: list                       # List of Syllable

    # Timing
    time: Optional[TimeSpan] = None

    # Phrase boundaries (with confidence)
    starts_mid_bar: bool = False          # Enjambment from previous
    ends_mid_bar: bool = False            # Enjambs into next
    breath_after: bool = False
    breath_confidence: float = 1.0

    # Linguistic
    is_complete_clause: bool = True
    clause_confidence: float = 1.0

    # Rhyme
    ends_with_rhyme: bool = False
    end_rhyme_group: Optional[str] = None

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.syllables)

    @property
    def bar_span(self) -> tuple:
        """Which bars does this phrase span?"""
        if not self.syllables:
            return (0, 0)
        bars = [s.time.start.bar for s in self.syllables if s.time and s.time.start.bar]
        if not bars:
            return (0, 0)
        return (min(bars), max(bars))


# =============================================================================
# BAR - MUSICAL STRUCTURE (separate from phrase content)
# =============================================================================

@dataclass
class Bar:
    """
    One bar of 4/4 time.
    Contains references to syllables/phrases, not copies.
    Phrases may span multiple bars (enjambment).
    """
    number: int
    time: Optional[TimeSpan] = None

    # Content (IDs, not objects - avoids duplication)
    syllable_ids: list = field(default_factory=list)
    phrase_ids: list = field(default_factory=list)  # Phrases active in this bar

    # Grid representations (multiple grids available)
    grid_16: list = field(default_factory=lambda: [None] * 16)
    grid_noct: list = field(default_factory=lambda: [None] * 18)
    grid_32: list = field(default_factory=lambda: [None] * 32)

    # Density (computed)
    syllable_count: int = 0
    syllable_density: float = 0.0         # Per beat

    # Rest analysis
    rest_positions: list = field(default_factory=list)  # Grid positions with no syllable
    longest_rest_beats: float = 0.0


# =============================================================================
# RHYME - WITH CONFIDENCE SCORES
# =============================================================================

@dataclass
class RhymeGroup:
    """A group of rhyming syllables"""
    id: str
    rhyme_type: RhymeType
    syllable_ids: list

    # Phonetic basis
    shared_sound: str                     # The common rime: "AE T"

    # Confidence
    confidence: float = 1.0               # Average pair confidence
    min_pair_confidence: float = 1.0      # Weakest link

    # Pattern analysis
    positions: list = field(default_factory=list)  # (bar, beat) tuples
    is_end_rhyme: bool = False
    is_internal: bool = False
    span_bars: int = 0


@dataclass
class RhymeScheme:
    """Complete rhyme analysis"""
    groups: list                          # List of RhymeGroup

    # Metrics (with confidence)
    rhyme_density: float = 0.0
    rhyme_density_confidence: float = 1.0

    multisyllabic_ratio: float = 0.0
    internal_ratio: float = 0.0

    # Pattern
    end_rhyme_pattern: str = ""           # "AABB", "ABAB", etc
    pattern_confidence: float = 1.0


# =============================================================================
# FLOW / MICROTIMING
# =============================================================================

@dataclass
class MicrotimingProfile:
    """
    How the rapper relates to the beat.
    PRESERVES raw data, provides analysis.
    """
    # Raw deviations (never throw these away)
    all_deviations_ms: list = field(default_factory=list)

    # Summary stats
    mean_deviation_ms: float = 0.0        # + = behind (laid back), - = ahead (rushing)
    std_deviation_ms: float = 0.0

    # Per-beat tendency
    beat_deviations: dict = field(default_factory=dict)  # {1: -5.2, 2: 3.1, ...}

    # Style classification (with confidence)
    style: str = "on-beat"                # "on-beat", "laid-back", "rushing", "variable"
    style_confidence: float = 1.0

    # Swing analysis
    swing_ratio: float = 0.5              # 0.5 = straight, 0.67 = swung
    swing_confidence: float = 1.0


@dataclass
class FlowPattern:
    """A detected rhythmic pattern"""
    id: str
    pattern: list                         # Relative durations or grid positions
    occurrences: list                     # TimeSpans where it appears

    confidence: float = 1.0

    # Classification
    is_syncopated: bool = False
    is_on_beat: bool = True
    complexity_score: float = 0.0


# =============================================================================
# BEAT / INSTRUMENTAL
# =============================================================================

@dataclass
class BeatAnalysis:
    """Analysis of the instrumental"""
    bpm: float
    bpm_confidence: float = 1.0

    time_signature: tuple = (4, 4)
    key: Optional[str] = None
    key_confidence: float = 0.0

    # Beat positions (in seconds)
    beat_times: list = field(default_factory=list)
    downbeat_times: list = field(default_factory=list)

    # Drum analysis
    has_drums: bool = True
    drum_pattern_type: str = "boom_bap"   # "boom_bap", "trap", "drill", etc

    total_bars: int = 0


# =============================================================================
# MAIN OUTPUT - FLEXIBLE, MULTI-LAYER
# =============================================================================

@dataclass
class RapAnalysis:
    """
    The complete analysis output.

    Key design choices:
    - Syllables are the source of truth
    - Phrases are separate from bars (enjambment OK)
    - Multiple grid representations available
    - Everything has confidence scores
    - Genre-aware
    """
    # Metadata
    id: str
    title: Optional[str] = None
    artist: Optional[str] = None
    duration_seconds: float = 0.0
    analyzed_at: str = ""

    # Genre context (affects interpretation)
    genre: Genre = Genre.CLASSIC_EAST
    genre_confidence: float = 0.5

    # Beat
    beat: Optional[BeatAnalysis] = None

    # Content - SYLLABLES ARE SOURCE OF TRUTH
    syllables: list = field(default_factory=list)  # All Syllable objects

    # Organizational layers (reference syllables by ID)
    phrases: list = field(default_factory=list)    # Independent from bars
    bars: list = field(default_factory=list)       # Musical structure

    # Word-level (reconstructed from syllables)
    words: list = field(default_factory=list)

    # Analysis results
    rhyme_scheme: Optional[RhymeScheme] = None
    microtiming: Optional[MicrotimingProfile] = None
    flow_patterns: list = field(default_factory=list)

    # Vocal style summary
    dominant_vocal_style: VocalStyle = VocalStyle.RAP_PERCUSSIVE
    melodicity_score: float = 0.0         # 0-1 average
    style_variance: float = 0.0           # How much it changes

    # Summary metrics
    total_syllables: int = 0
    average_syllables_per_bar: float = 0.0
    vocabulary_size: int = 0

    # Quality metrics (how confident are we in this analysis?)
    overall_confidence: float = 1.0
    transcription_confidence: float = 1.0
    alignment_confidence: float = 1.0

    def get_syllables_in_bar(self, bar_num: int) -> list:
        """Get all syllables in a specific bar"""
        return [s for s in self.syllables
                if s.time and s.time.start.bar == bar_num]

    def get_phrases_in_bar(self, bar_num: int) -> list:
        """Get phrases active in a bar (may span multiple bars)"""
        result = []
        for phrase in self.phrases:
            start_bar, end_bar = phrase.bar_span
            if start_bar <= bar_num <= end_bar:
                result.append(phrase)
        return result

    def syllables_by_grid(self, grid_type: GridType = GridType.SIXTEENTH) -> dict:
        """Get syllables organized by grid position per bar"""
        result = {}
        for bar in self.bars:
            if grid_type == GridType.SIXTEENTH:
                result[bar.number] = bar.grid_16
            elif grid_type == GridType.NOCTUPLET:
                result[bar.number] = bar.grid_noct
            elif grid_type == GridType.THIRTYSECOND:
                result[bar.number] = bar.grid_32
        return result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def syllables_to_text(syllables: list, join_char: str = " ") -> str:
    """Convert syllable list to readable text"""
    return join_char.join(s.text for s in syllables)


def detect_enjambment(phrases: list, bars: list) -> list:
    """Find phrases that cross bar boundaries"""
    enjambed = []
    for phrase in phrases:
        start_bar, end_bar = phrase.bar_span
        if start_bar != end_bar:
            enjambed.append({
                'phrase_id': phrase.id,
                'text': phrase.text,
                'bars': (start_bar, end_bar),
                'bar_count': end_bar - start_bar + 1
            })
    return enjambed


def classify_vocal_style(syllables: list) -> tuple:
    """
    Determine overall vocal style from syllables.
    Returns (dominant_style, melodicity_score, variance)
    """
    if not syllables:
        return VocalStyle.RAP_PERCUSSIVE, 0.0, 0.0

    melodicities = [s.melodicity_score for s in syllables]
    avg = sum(melodicities) / len(melodicities)
    variance = sum((m - avg) ** 2 for m in melodicities) / len(melodicities)

    if avg < 0.2:
        style = VocalStyle.RAP_PERCUSSIVE
    elif avg < 0.4:
        style = VocalStyle.RAP_MELODIC
    elif avg < 0.7:
        style = VocalStyle.SING_RAP
    else:
        style = VocalStyle.SINGING

    return style, avg, variance
