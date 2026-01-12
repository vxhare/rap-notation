"""
grid_alignment.py - Rhythmic Grid Alignment for Rap Analysis

Aligns syllables to rhythmic grids and calculates microtiming deviation.

Supports:
- 16th note grid (standard hip-hop quantization)
- Noctuplet grid (Connor's 9-per-half-bar system)
- Grid-fit scoring
- Microtiming analysis
- Swing/phase/tempo optimization
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import itertools


# =============================================================================
# CONSTANTS
# =============================================================================

class GridType(Enum):
    SIXTEENTH = 16
    NOCTUPLET = 18
    THIRTYSECOND = 32


GRID_POSITIONS = {
    GridType.SIXTEENTH: [i * 0.25 for i in range(16)],
    GridType.NOCTUPLET: [
        *(i * (2.0 / 9) for i in range(9)),
        *(2 + i * (2.0 / 9) for i in range(9))
    ],
    GridType.THIRTYSECOND: [i * 0.125 for i in range(32)],
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GridPosition:
    """A syllable's position on a rhythmic grid"""
    grid_type: GridType
    bar: int
    slot: int
    beat: float
    deviation_ms: float
    deviation_beats: float

    @property
    def is_on_beat(self) -> bool:
        if self.grid_type == GridType.SIXTEENTH:
            return self.slot % 4 == 0
        elif self.grid_type == GridType.NOCTUPLET:
            return self.slot in (0, 9)
        return False


@dataclass
class AlignedSyllable:
    """A syllable with grid alignment data"""
    id: str
    text: str
    start_seconds: float
    end_seconds: float
    bar: int
    beat: float
    grid_16: GridPosition
    grid_noct: GridPosition
    best_grid: GridType
    best_deviation_ms: float


@dataclass
class GridFitScore:
    """How well a passage fits a particular grid"""
    grid_type: GridType
    mean_deviation_ms: float
    std_deviation_ms: float
    max_deviation_ms: float
    fit_score: float
    on_grid_count: int
    off_grid_count: int
    on_grid_ratio: float


@dataclass
class MicrotimingProfile:
    """How a rapper relates to the beat"""
    mean_deviation_ms: float
    std_deviation_ms: float
    beat_deviations: dict
    style: str
    swing_ratio: float = 1.0
    phase_shift: float = 0.0
    tempo_adjust: float = 1.0


# =============================================================================
# GRID ALIGNER
# =============================================================================

class GridAligner:
    """Aligns syllables to rhythmic grids."""

    def __init__(
        self,
        bpm: float,
        bar_one_start: float = 0.0,
        on_grid_threshold_ms: float = 30.0
    ):
        self.bpm = bpm
        self.bar_one_start = bar_one_start
        self.on_grid_threshold_ms = on_grid_threshold_ms
        self.beat_duration = 60.0 / bpm
        self.bar_duration = self.beat_duration * 4
        self.ms_per_beat = self.beat_duration * 1000

    def align_syllables(self, syllables: list) -> list:
        """Align a list of syllables to both grids."""
        aligned = []

        for syl in syllables:
            start = syl['start']
            end = syl.get('end', start + 0.1)
            bar, beat = self._time_to_bar_beat(start)

            grid_16 = self._align_to_grid(bar, beat, GridType.SIXTEENTH)
            grid_noct = self._align_to_grid(bar, beat, GridType.NOCTUPLET)

            if abs(grid_16.deviation_ms) <= abs(grid_noct.deviation_ms):
                best_grid = GridType.SIXTEENTH
                best_dev = grid_16.deviation_ms
            else:
                best_grid = GridType.NOCTUPLET
                best_dev = grid_noct.deviation_ms

            aligned.append(AlignedSyllable(
                id=syl.get('id', f"syl_{len(aligned)}"),
                text=syl.get('text', ''),
                start_seconds=start,
                end_seconds=end,
                bar=bar,
                beat=beat,
                grid_16=grid_16,
                grid_noct=grid_noct,
                best_grid=best_grid,
                best_deviation_ms=best_dev
            ))

        return aligned

    def _time_to_bar_beat(self, time_seconds: float) -> tuple:
        """Convert absolute time to bar and beat"""
        time_from_start = time_seconds - self.bar_one_start
        total_beats = time_from_start / self.beat_duration
        bar = int(total_beats // 4) + 1
        beat = (total_beats % 4) + 1
        return bar, beat

    def _align_to_grid(self, bar: int, beat: float, grid_type: GridType) -> GridPosition:
        """Find the nearest grid position"""
        grid_positions = GRID_POSITIONS[grid_type]
        beat_in_bar = beat - 1.0

        min_dist = float('inf')
        nearest_slot = 0

        for slot, grid_beat in enumerate(grid_positions):
            dist = abs(beat_in_bar - grid_beat)
            wrapped_dist = min(dist, 4.0 - dist)
            if wrapped_dist < min_dist:
                min_dist = wrapped_dist
                nearest_slot = slot

        grid_beat = grid_positions[nearest_slot]
        deviation_beats = beat_in_bar - grid_beat

        if deviation_beats > 2.0:
            deviation_beats -= 4.0
        elif deviation_beats < -2.0:
            deviation_beats += 4.0

        deviation_ms = deviation_beats * self.ms_per_beat

        return GridPosition(
            grid_type=grid_type,
            bar=bar,
            slot=nearest_slot,
            beat=beat,
            deviation_ms=deviation_ms,
            deviation_beats=deviation_beats
        )

    def score_grid_fit(self, aligned_syllables: list, grid_type: GridType) -> GridFitScore:
        """Score how well syllables fit a particular grid."""
        if not aligned_syllables:
            return GridFitScore(
                grid_type=grid_type, mean_deviation_ms=0, std_deviation_ms=0,
                max_deviation_ms=0, fit_score=0, on_grid_count=0,
                off_grid_count=0, on_grid_ratio=0
            )

        if grid_type == GridType.SIXTEENTH:
            deviations = [abs(s.grid_16.deviation_ms) for s in aligned_syllables]
        else:
            deviations = [abs(s.grid_noct.deviation_ms) for s in aligned_syllables]

        deviations = np.array(deviations)
        mean_dev = np.mean(deviations)
        std_dev = np.std(deviations)
        max_dev = np.max(deviations)

        on_grid = np.sum(deviations <= self.on_grid_threshold_ms)
        off_grid = len(deviations) - on_grid
        on_grid_ratio = on_grid / len(deviations)

        scale = 50.0
        fit_score = np.exp(-mean_dev / scale) * on_grid_ratio

        return GridFitScore(
            grid_type=grid_type,
            mean_deviation_ms=float(mean_dev),
            std_deviation_ms=float(std_dev),
            max_deviation_ms=float(max_dev),
            fit_score=float(fit_score),
            on_grid_count=int(on_grid),
            off_grid_count=int(off_grid),
            on_grid_ratio=float(on_grid_ratio)
        )

    def compare_grids(self, aligned_syllables: list) -> dict:
        """Compare how well syllables fit each grid type."""
        score_16 = self.score_grid_fit(aligned_syllables, GridType.SIXTEENTH)
        score_noct = self.score_grid_fit(aligned_syllables, GridType.NOCTUPLET)

        if score_16.fit_score >= score_noct.fit_score:
            recommended = GridType.SIXTEENTH
            confidence = score_16.fit_score - score_noct.fit_score
        else:
            recommended = GridType.NOCTUPLET
            confidence = score_noct.fit_score - score_16.fit_score

        return {
            "sixteenth": {
                "fit_score": score_16.fit_score,
                "mean_deviation_ms": score_16.mean_deviation_ms,
                "on_grid_ratio": score_16.on_grid_ratio,
            },
            "noctuplet": {
                "fit_score": score_noct.fit_score,
                "mean_deviation_ms": score_noct.mean_deviation_ms,
                "on_grid_ratio": score_noct.on_grid_ratio,
            },
            "recommended": recommended.name,
            "confidence": confidence,
        }

    def analyze_microtiming(
        self,
        aligned_syllables: list,
        grid_type: GridType = GridType.SIXTEENTH
    ) -> MicrotimingProfile:
        """Analyze the rapper's microtiming tendencies."""
        if not aligned_syllables:
            return MicrotimingProfile(
                mean_deviation_ms=0, std_deviation_ms=0,
                beat_deviations={1: 0, 2: 0, 3: 0, 4: 0}, style="unknown"
            )

        if grid_type == GridType.SIXTEENTH:
            deviations = [s.grid_16.deviation_ms for s in aligned_syllables]
            beats = [int(s.beat) for s in aligned_syllables]
        else:
            deviations = [s.grid_noct.deviation_ms for s in aligned_syllables]
            beats = [int(s.beat) for s in aligned_syllables]

        mean_dev = np.mean(deviations)
        std_dev = np.std(deviations)

        beat_devs = {1: [], 2: [], 3: [], 4: []}
        for beat, dev in zip(beats, deviations):
            if 1 <= beat <= 4:
                beat_devs[beat].append(dev)

        beat_deviations = {
            b: float(np.mean(devs)) if devs else 0.0
            for b, devs in beat_devs.items()
        }

        if abs(mean_dev) < 10 and std_dev < 20:
            style = "on-beat"
        elif mean_dev > 15:
            style = "laid-back"
        elif mean_dev < -15:
            style = "rushing"
        elif std_dev > 40:
            style = "variable"
        else:
            style = "on-beat"

        return MicrotimingProfile(
            mean_deviation_ms=float(mean_dev),
            std_deviation_ms=float(std_dev),
            beat_deviations=beat_deviations,
            style=style
        )


# =============================================================================
# OPTIMIZER
# =============================================================================

class MicrotimingOptimizer:
    """Optimizes swing, phase, and tempo to minimize deviation."""

    def __init__(
        self,
        swing_range: tuple = (0.5, 1.5),
        phase_range: tuple = (-30, 30),
        tempo_range: tuple = (0.95, 1.05),
        grid_resolution: int = 10
    ):
        self.swing_range = swing_range
        self.phase_range = phase_range
        self.tempo_range = tempo_range
        self.grid_resolution = grid_resolution

    def optimize(self, syllables: list, bpm: float) -> dict:
        """Find optimal swing/phase/tempo parameters."""
        if not syllables:
            return {"error": "No syllables provided"}

        positions = [(s.beat, s.grid_16.deviation_ms) for s in syllables]
        original_dev = np.mean([abs(p[1]) for p in positions])

        swings = np.linspace(*self.swing_range, self.grid_resolution)
        phases = np.linspace(*self.phase_range, self.grid_resolution)
        tempos = np.linspace(*self.tempo_range, self.grid_resolution)

        best_dev = original_dev
        best_params = {"swing": 1.0, "phase": 0.0, "tempo": 1.0}

        for swing, phase, tempo in itertools.product(swings, phases, tempos):
            adjusted_dev = self._calculate_adjusted_deviation(
                positions, swing, phase, tempo, bpm
            )
            if adjusted_dev < best_dev - 0.5:
                best_dev = adjusted_dev
                best_params = {"swing": swing, "phase": phase, "tempo": tempo}

        return {
            "original_deviation_ms": float(original_dev),
            "optimized_deviation_ms": float(best_dev),
            "improvement_ms": float(original_dev - best_dev),
            "parameters": best_params,
        }

    def _calculate_adjusted_deviation(
        self, positions: list, swing: float, phase: float, tempo: float, bpm: float
    ) -> float:
        ms_per_beat = 60000 / bpm
        adjusted_deviations = []

        for beat, original_dev in positions:
            adjusted_dev = original_dev * tempo
            phase_ms = phase * (ms_per_beat / 60)
            adjusted_dev -= phase_ms

            beat_fraction = beat % 1.0
            if 0.4 < beat_fraction < 0.6:
                swing_adjustment = (swing - 1.0) * ms_per_beat * 0.25
                adjusted_dev -= swing_adjustment

            adjusted_deviations.append(abs(adjusted_dev))

        return np.mean(adjusted_deviations)


# =============================================================================
# BAR GRID BUILDER
# =============================================================================

def build_bar_grid(aligned_syllables: list, grid_type: GridType = GridType.SIXTEENTH) -> list:
    """Build bar-by-bar grid representation for visualization."""
    if not aligned_syllables:
        return []

    grid_size = grid_type.value
    bars = {}

    for syl in aligned_syllables:
        bar = syl.bar
        if bar not in bars:
            bars[bar] = [None] * grid_size

        if grid_type == GridType.SIXTEENTH:
            slot = syl.grid_16.slot
            dev = syl.grid_16.deviation_ms
        else:
            slot = syl.grid_noct.slot
            dev = syl.grid_noct.deviation_ms

        if 0 <= slot < grid_size:
            bars[bar][slot] = {"id": syl.id, "text": syl.text, "deviation_ms": dev}

    result = []
    for bar_num in sorted(bars.keys()):
        slots = []
        for pos, content in enumerate(bars[bar_num]):
            if content:
                slots.append({
                    "position": pos,
                    "text": content["text"],
                    "deviation_ms": content["deviation_ms"],
                    "syllable_id": content["id"]
                })
            else:
                slots.append({"position": pos, "text": None, "deviation_ms": None, "syllable_id": None})

        result.append({"bar": bar_num, "slots": slots, "grid_type": grid_type.name})

    return result


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def align_and_analyze(syllables: list, bpm: float, bar_one_start: float = 0.0) -> dict:
    """Complete alignment and analysis in one call."""
    aligner = GridAligner(bpm, bar_one_start)
    optimizer = MicrotimingOptimizer()

    aligned = aligner.align_syllables(syllables)
    grid_comparison = aligner.compare_grids(aligned)
    microtiming = aligner.analyze_microtiming(aligned)
    optimization = optimizer.optimize(aligned, bpm)
    grid_16 = build_bar_grid(aligned, GridType.SIXTEENTH)
    grid_noct = build_bar_grid(aligned, GridType.NOCTUPLET)

    return {
        "syllables": [
            {
                "id": s.id, "text": s.text, "bar": s.bar, "beat": s.beat,
                "grid_16_slot": s.grid_16.slot,
                "grid_16_deviation_ms": s.grid_16.deviation_ms,
                "grid_noct_slot": s.grid_noct.slot,
                "grid_noct_deviation_ms": s.grid_noct.deviation_ms,
                "best_grid": s.best_grid.name,
            }
            for s in aligned
        ],
        "grid_comparison": grid_comparison,
        "microtiming": {
            "mean_deviation_ms": microtiming.mean_deviation_ms,
            "std_deviation_ms": microtiming.std_deviation_ms,
            "style": microtiming.style,
            "beat_deviations": microtiming.beat_deviations,
        },
        "optimization": optimization,
        "bar_grids": {"sixteenth": grid_16, "noctuplet": grid_noct},
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Grid Alignment Test\n")
    print("=" * 60)

    bpm = 90
    test_syllables = [
        {"id": "s1", "text": "I", "start": 0.0, "end": 0.15},
        {"id": "s2", "text": "got", "start": 0.17, "end": 0.32},
        {"id": "s3", "text": "the", "start": 0.33, "end": 0.48},
        {"id": "s4", "text": "strap", "start": 0.50, "end": 0.65},
        {"id": "s5", "text": "I", "start": 0.70, "end": 0.80},
        {"id": "s6", "text": "got", "start": 0.85, "end": 0.95},
        {"id": "s7", "text": "the", "start": 1.00, "end": 1.10},
        {"id": "s8", "text": "MAC", "start": 1.15, "end": 1.30},
    ]

    result = align_and_analyze(test_syllables, bpm)

    print(f"BPM: {bpm}")
    print(f"Recommended grid: {result['grid_comparison']['recommended']}")
    print(f"Microtiming style: {result['microtiming']['style']}")
    print(f"Mean deviation: {result['microtiming']['mean_deviation_ms']:.1f}ms")
    print()

    print("Syllable alignment:")
    print(f"{'TEXT':<8} {'BAR':<4} {'BEAT':<6} {'SLOT':<5} {'DEV(ms)':<10}")
    print("-" * 40)

    for syl in result['syllables']:
        print(f"{syl['text']:<8} {syl['bar']:<4} {syl['beat']:<6.2f} "
              f"{syl['grid_16_slot']:<5} {syl['grid_16_deviation_ms']:<10.1f}")
