"""
flow_metrics.py - Flow Analysis Metrics for Rap

Quantitative metrics for analyzing rap flow.

Includes:
- Rhyme entropy (Shannon entropy on inter-rhyme intervals)
- Groove complexity (rhythmic unpredictability)
- Syllable density metrics
- Flow variance over time
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter
import math


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RhymeEntropyResult:
    """Results from rhyme entropy analysis"""
    entropy: float
    entropy_raw: float
    num_rhymes: int
    num_intervals: int
    interval_distribution: dict
    pattern_type: str
    dominant_interval: Optional[float]
    entropy_mod4: float


@dataclass
class GrooveComplexityResult:
    """Results from groove complexity analysis"""
    complexity: float
    syncopation: float
    density_variance: float
    rest_unpredictability: float
    bar_complexities: list
    style: str


@dataclass
class FlowProfile:
    """Complete flow analysis for a verse"""
    syllables_per_bar: float
    syllables_per_beat: float
    density_variance: float
    rhyme_density: float
    rhyme_entropy: RhymeEntropyResult
    groove_complexity: GrooveComplexityResult
    syncopation_ratio: float
    flow_variance: float
    percentiles: dict = field(default_factory=dict)


# =============================================================================
# RHYME ENTROPY
# =============================================================================

class RhymeEntropyAnalyzer:
    """Calculate Shannon entropy of inter-rhyme intervals."""

    def __init__(self, max_interval: float = 8.0, use_stressed_only: bool = True):
        self.max_interval = max_interval
        self.use_stressed_only = use_stressed_only

    def analyze(self, syllables: list, rhyme_groups: Optional[dict] = None) -> RhymeEntropyResult:
        """Calculate rhyme entropy."""
        syl_lookup = {s['id']: s for s in syllables}

        if rhyme_groups is None:
            rhyme_groups = self._extract_rhyme_groups(syllables)

        if not rhyme_groups:
            return self._empty_result()

        all_iris = []

        for rhyme_class, syl_ids in rhyme_groups.items():
            positions = []
            for syl_id in syl_ids:
                syl = syl_lookup.get(syl_id)
                if syl:
                    if self.use_stressed_only and not syl.get('stressed', True):
                        continue
                    bar = syl.get('bar', 1)
                    beat = syl.get('beat', 1.0)
                    abs_beat = (bar - 1) * 4 + beat
                    positions.append(abs_beat)

            if len(positions) < 2:
                continue

            positions.sort()
            for i in range(1, len(positions)):
                iri = positions[i] - positions[i-1]
                if iri <= self.max_interval:
                    all_iris.append(iri)

        if not all_iris:
            return self._empty_result()

        entropy, entropy_raw, distribution = self._calculate_entropy(all_iris)
        iris_mod4 = [iri % 4 for iri in all_iris]
        entropy_mod4, _, _ = self._calculate_entropy(iris_mod4)

        dominant = max(distribution, key=distribution.get) if distribution else None

        if entropy < 0.3:
            pattern_type = "regular"
        elif entropy < 0.7:
            pattern_type = "varied"
        else:
            pattern_type = "unpredictable"

        return RhymeEntropyResult(
            entropy=entropy,
            entropy_raw=entropy_raw,
            num_rhymes=sum(len(ids) for ids in rhyme_groups.values()),
            num_intervals=len(all_iris),
            interval_distribution=distribution,
            pattern_type=pattern_type,
            dominant_interval=dominant,
            entropy_mod4=entropy_mod4
        )

    def _extract_rhyme_groups(self, syllables: list) -> dict:
        groups = {}
        for syl in syllables:
            rhyme_class = syl.get('rhyme_class')
            if rhyme_class:
                if rhyme_class not in groups:
                    groups[rhyme_class] = []
                groups[rhyme_class].append(syl['id'])
        return groups

    def _calculate_entropy(self, values: list) -> tuple:
        if not values:
            return 0.0, 0.0, {}

        quantized = [round(v * 4) / 4 for v in values]
        counts = Counter(quantized)
        total = len(quantized)
        probs = [count / total for count in counts.values()]

        entropy_raw = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1
        entropy_normalized = entropy_raw / max_entropy if max_entropy > 0 else 0

        return entropy_normalized, entropy_raw, dict(counts)

    def _empty_result(self) -> RhymeEntropyResult:
        return RhymeEntropyResult(
            entropy=0.0, entropy_raw=0.0, num_rhymes=0, num_intervals=0,
            interval_distribution={}, pattern_type="none",
            dominant_interval=None, entropy_mod4=0.0
        )


# =============================================================================
# GROOVE COMPLEXITY
# =============================================================================

class GrooveComplexityAnalyzer:
    """Analyze rhythmic complexity of flow."""

    def __init__(self, grid_resolution: int = 16, window_size: int = 4):
        self.grid_resolution = grid_resolution
        self.window_size = window_size

    def analyze(self, syllables: list) -> GrooveComplexityResult:
        if not syllables:
            return self._empty_result()

        bar_grids = self._build_grids(syllables)
        if not bar_grids:
            return self._empty_result()

        syncopation = self._calculate_syncopation(syllables)
        density_variance = self._calculate_density_variance(bar_grids)
        rest_unpred = self._calculate_rest_unpredictability(bar_grids)
        bar_complexities = [self._bar_complexity(grid) for grid in bar_grids.values()]

        complexity = (
            syncopation * 0.35 +
            density_variance * 0.25 +
            rest_unpred * 0.25 +
            np.mean(bar_complexities) * 0.15
        )

        if complexity < 0.3:
            style = "simple"
        elif complexity < 0.6:
            style = "moderate"
        else:
            style = "complex"

        return GrooveComplexityResult(
            complexity=float(complexity),
            syncopation=float(syncopation),
            density_variance=float(density_variance),
            rest_unpredictability=float(rest_unpred),
            bar_complexities=bar_complexities,
            style=style
        )

    def _build_grids(self, syllables: list) -> dict:
        bar_grids = {}
        for syl in syllables:
            bar = syl.get('bar', 1)
            if 'grid_slot' in syl:
                slot = syl['grid_slot']
            elif 'beat' in syl:
                beat = syl['beat']
                beat_in_bar = beat - 1 if beat >= 1 else beat
                slot = int(beat_in_bar * (self.grid_resolution / 4))
            else:
                continue

            if bar not in bar_grids:
                bar_grids[bar] = [0] * self.grid_resolution
            if 0 <= slot < self.grid_resolution:
                bar_grids[bar][slot] = 1

        return bar_grids

    def _calculate_syncopation(self, syllables: list) -> float:
        if not syllables:
            return 0.0

        weak_count = 0
        total = 0
        strong_slots = {0, 4, 8, 12}
        medium_slots = {2, 6, 10, 14}

        for syl in syllables:
            if 'grid_slot' in syl:
                slot = syl['grid_slot'] % self.grid_resolution
            elif 'beat' in syl:
                beat = syl['beat']
                beat_fraction = (beat - 1) % 1
                slot = int(beat_fraction * 4) + int((beat - 1) % 4) * 4
                slot = slot % self.grid_resolution
            else:
                continue

            total += 1
            if slot not in strong_slots and slot not in medium_slots:
                weak_count += 1

        return weak_count / total if total > 0 else 0.0

    def _calculate_density_variance(self, bar_grids: dict) -> float:
        if not bar_grids:
            return 0.0

        densities = [sum(grid) for grid in bar_grids.values()]
        if len(densities) < 2:
            return 0.0

        mean_density = np.mean(densities)
        if mean_density == 0:
            return 0.0

        return min(1.0, np.std(densities) / mean_density)

    def _calculate_rest_unpredictability(self, bar_grids: dict) -> float:
        rest_lengths = []
        for grid in bar_grids.values():
            current_rest = 0
            for slot in grid:
                if slot == 0:
                    current_rest += 1
                else:
                    if current_rest > 0:
                        rest_lengths.append(current_rest)
                    current_rest = 0
            if current_rest > 0:
                rest_lengths.append(current_rest)

        if not rest_lengths:
            return 0.0

        counts = Counter(rest_lengths)
        total = len(rest_lengths)
        probs = [c / total for c in counts.values()]

        if len(probs) <= 1:
            return 0.0

        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(len(counts))

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _bar_complexity(self, grid: list) -> float:
        if not grid or sum(grid) == 0:
            return 0.0

        transitions = sum(1 for i in range(1, len(grid)) if grid[i] != grid[i-1])
        max_transitions = len(grid) - 1
        transition_ratio = transitions / max_transitions if max_transitions > 0 else 0

        syllable_positions = [i for i, v in enumerate(grid) if v == 1]
        if len(syllable_positions) < 2:
            return transition_ratio * 0.5

        ideal_spacing = len(grid) / len(syllable_positions)
        spacings = [
            syllable_positions[i+1] - syllable_positions[i]
            for i in range(len(syllable_positions) - 1)
        ]
        spacing_variance = np.std(spacings) / ideal_spacing if ideal_spacing > 0 else 0

        return (transition_ratio * 0.5 + min(1.0, spacing_variance) * 0.5)

    def _empty_result(self) -> GrooveComplexityResult:
        return GrooveComplexityResult(
            complexity=0.0, syncopation=0.0, density_variance=0.0,
            rest_unpredictability=0.0, bar_complexities=[], style="none"
        )


# =============================================================================
# DENSITY ANALYZER
# =============================================================================

class DensityAnalyzer:
    """Analyze syllable density patterns."""

    def analyze(self, syllables: list, total_bars: Optional[int] = None) -> dict:
        if not syllables:
            return self._empty_result()

        bars = [s.get('bar', 1) for s in syllables]
        min_bar = min(bars)
        max_bar = max(bars) if total_bars is None else total_bars

        bar_counts = Counter(bars)
        densities = [bar_counts.get(b, 0) for b in range(min_bar, max_bar + 1)]

        total_syllables = len(syllables)
        num_bars = max_bar - min_bar + 1

        syllables_per_bar = total_syllables / num_bars if num_bars > 0 else 0
        syllables_per_beat = syllables_per_bar / 4

        density_variance = float(np.std(densities)) if densities else 0
        density_cv = density_variance / np.mean(densities) if np.mean(densities) > 0 else 0

        max_density = max(densities) if densities else 0
        peak_bars = [b + min_bar for b, d in enumerate(densities) if d == max_density]
        rest_bars = [b + min_bar for b, d in enumerate(densities) if d <= 1]

        if len(densities) >= 4:
            first_half = np.mean(densities[:len(densities)//2])
            second_half = np.mean(densities[len(densities)//2:])
            if second_half > first_half * 1.2:
                trend = "increasing"
            elif second_half < first_half * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "total_syllables": total_syllables,
            "total_bars": num_bars,
            "syllables_per_bar": syllables_per_bar,
            "syllables_per_beat": syllables_per_beat,
            "density_variance": density_variance,
            "density_cv": density_cv,
            "max_density": max_density,
            "min_density": min(densities) if densities else 0,
            "peak_bars": peak_bars,
            "rest_bars": rest_bars,
            "trend": trend,
            "bar_densities": densities
        }

    def _empty_result(self) -> dict:
        return {
            "total_syllables": 0, "total_bars": 0, "syllables_per_bar": 0,
            "syllables_per_beat": 0, "density_variance": 0, "density_cv": 0,
            "max_density": 0, "min_density": 0, "peak_bars": [], "rest_bars": [],
            "trend": "none", "bar_densities": []
        }


# =============================================================================
# FLOW VARIANCE
# =============================================================================

class FlowVarianceAnalyzer:
    """Measure how much the flow changes throughout a verse."""

    def __init__(self, window_size: int = 4):
        self.window_size = window_size

    def analyze(self, syllables: list) -> dict:
        if not syllables:
            return {"variance": 0, "changes": [], "style": "none"}

        bars = {}
        for syl in syllables:
            bar = syl.get('bar', 1)
            if bar not in bars:
                bars[bar] = []
            bars[bar].append(syl)

        if len(bars) < self.window_size * 2:
            return {"variance": 0, "changes": [], "style": "consistent"}

        bar_nums = sorted(bars.keys())
        windows = []

        for i in range(0, len(bar_nums) - self.window_size + 1, self.window_size):
            window_bars = bar_nums[i:i + self.window_size]
            window_syls = []
            for b in window_bars:
                window_syls.extend(bars[b])

            features = self._window_features(window_syls)
            windows.append(features)

        if len(windows) < 2:
            return {"variance": 0, "changes": [], "style": "consistent"}

        changes = []
        for i in range(1, len(windows)):
            change = self._feature_distance(windows[i-1], windows[i])
            changes.append({"window": i, "change_magnitude": change})

        variance = np.mean([c["change_magnitude"] for c in changes])

        if variance < 0.2:
            style = "consistent"
        elif variance < 0.5:
            style = "moderate_variation"
        else:
            style = "high_variation"

        return {"variance": float(variance), "changes": changes, "style": style, "num_windows": len(windows)}

    def _window_features(self, syllables: list) -> dict:
        if not syllables:
            return {"density": 0, "syncopation": 0, "spread": 0}

        bars_in_window = set(s.get('bar', 1) for s in syllables)
        density = len(syllables) / len(bars_in_window) if bars_in_window else 0

        beats = [s.get('beat', 1) for s in syllables]
        on_beat_count = sum(1 for b in beats if b % 1 < 0.1 or b % 1 > 0.9)
        syncopation = 1 - (on_beat_count / len(beats)) if beats else 0

        beat_fractions = [b % 1 for b in beats]
        spread = float(np.std(beat_fractions)) if len(beat_fractions) > 1 else 0

        return {"density": density, "syncopation": syncopation, "spread": spread}

    def _feature_distance(self, f1: dict, f2: dict) -> float:
        diff_density = abs(f1["density"] - f2["density"]) / max(f1["density"], f2["density"], 1)
        diff_sync = abs(f1["syncopation"] - f2["syncopation"])
        diff_spread = abs(f1["spread"] - f2["spread"])
        return (diff_density + diff_sync + diff_spread) / 3


# =============================================================================
# FLOW PROFILER
# =============================================================================

class FlowProfiler:
    """Generate complete flow profile for a verse."""

    def __init__(self):
        self.rhyme_analyzer = RhymeEntropyAnalyzer()
        self.groove_analyzer = GrooveComplexityAnalyzer()
        self.density_analyzer = DensityAnalyzer()
        self.variance_analyzer = FlowVarianceAnalyzer()

    def analyze(self, syllables: list, rhyme_groups: Optional[dict] = None) -> FlowProfile:
        density = self.density_analyzer.analyze(syllables)
        rhyme_entropy = self.rhyme_analyzer.analyze(syllables, rhyme_groups)
        groove = self.groove_analyzer.analyze(syllables)
        variance = self.variance_analyzer.analyze(syllables)

        rhyming_count = sum(1 for s in syllables if s.get('rhyme_class'))
        rhyme_density = rhyming_count / density["total_bars"] if density["total_bars"] > 0 else 0

        return FlowProfile(
            syllables_per_bar=density["syllables_per_bar"],
            syllables_per_beat=density["syllables_per_beat"],
            density_variance=density["density_variance"],
            rhyme_density=rhyme_density,
            rhyme_entropy=rhyme_entropy,
            groove_complexity=groove,
            syncopation_ratio=groove.syncopation,
            flow_variance=variance["variance"],
        )


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def analyze_flow(syllables: list, rhyme_groups: Optional[dict] = None) -> dict:
    """Complete flow analysis in one call."""
    profiler = FlowProfiler()
    profile = profiler.analyze(syllables, rhyme_groups)

    return {
        "density": {
            "syllables_per_bar": profile.syllables_per_bar,
            "syllables_per_beat": profile.syllables_per_beat,
            "variance": profile.density_variance,
        },
        "rhyme": {
            "density": profile.rhyme_density,
            "entropy": profile.rhyme_entropy.entropy,
            "pattern_type": profile.rhyme_entropy.pattern_type,
            "dominant_interval": profile.rhyme_entropy.dominant_interval,
            "num_rhymes": profile.rhyme_entropy.num_rhymes,
        },
        "rhythm": {
            "complexity": profile.groove_complexity.complexity,
            "syncopation": profile.syncopation_ratio,
            "style": profile.groove_complexity.style,
        },
        "variation": {
            "flow_variance": profile.flow_variance,
        }
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Flow Metrics Test\n")
    print("=" * 60)

    test_syllables = []

    bar1_beats = [1.0, 1.5, 2.0, 2.5, 3.0, 3.25, 3.5, 4.0]
    bar1_texts = ["I", "got", "the", "strap", "I", "got", "the", "MAC"]
    bar1_rhymes = [None, None, None, "A", None, None, None, "A"]

    for i, (beat, text, rhyme) in enumerate(zip(bar1_beats, bar1_texts, bar1_rhymes)):
        test_syllables.append({
            "id": f"s{i}", "text": text, "bar": 1, "beat": beat,
            "grid_slot": int((beat - 1) * 4), "rhyme_class": rhyme,
            "stressed": text in ["strap", "MAC"]
        })

    bar2_beats = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    bar2_texts = ["I'm", "in", "the", "back", "with", "the", "pack"]
    bar2_rhymes = [None, None, None, "A", None, None, "A"]

    for i, (beat, text, rhyme) in enumerate(zip(bar2_beats, bar2_texts, bar2_rhymes)):
        test_syllables.append({
            "id": f"s{len(test_syllables)}", "text": text, "bar": 2, "beat": beat,
            "grid_slot": int((beat - 1) * 4), "rhyme_class": rhyme,
            "stressed": text in ["back", "pack"]
        })

    result = analyze_flow(test_syllables)

    print("DENSITY")
    print(f"  Syllables/bar: {result['density']['syllables_per_bar']:.1f}")
    print(f"  Syllables/beat: {result['density']['syllables_per_beat']:.2f}")
    print()

    print("RHYME")
    print(f"  Rhyme density: {result['rhyme']['density']:.1f} per bar")
    print(f"  Rhyme entropy: {result['rhyme']['entropy']:.3f}")
    print(f"  Pattern type: {result['rhyme']['pattern_type']}")
    print()

    print("RHYTHM")
    print(f"  Groove complexity: {result['rhythm']['complexity']:.3f}")
    print(f"  Syncopation: {result['rhythm']['syncopation']:.1%}")
    print(f"  Style: {result['rhythm']['style']}")
