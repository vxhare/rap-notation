"""
Genre-specific configuration profiles.

Different genres have different:
- Expected syllable densities
- Typical rhyme patterns
- Microtiming tendencies
- Grid preferences
- Melodicity norms

ICP ≠ Kendrick ≠ Three 6 Mafia
"""

from dataclasses import dataclass
from typing import Optional
from models import Genre, GridType


@dataclass
class GenreProfile:
    """Configuration for a specific genre"""
    genre: Genre
    name: str
    description: str

    # Rhythm expectations
    typical_bpm_range: tuple = (80, 100)
    preferred_grid: GridType = GridType.SIXTEENTH
    allows_triplet_feel: bool = False
    typical_swing: float = 0.5            # 0.5 = straight

    # Density norms
    typical_syllables_per_bar: tuple = (8, 16)  # (low, high)
    typical_syllables_per_beat: tuple = (2, 4)

    # Rhyme expectations
    typical_rhyme_density: tuple = (1.0, 3.0)   # Rhymes per bar
    expect_multisyllabic: bool = True
    expect_internal_rhymes: bool = True
    end_rhyme_importance: float = 0.8     # How much to weight end rhymes

    # Microtiming
    typical_deviation_ms: tuple = (-20, 20)
    laid_back_tendency: float = 0.0       # + = tends laid back, - = rushing

    # Vocal style
    expected_melodicity: tuple = (0.0, 0.3)
    allows_singing: bool = False

    # Rhyme detection adjustments
    rhyme_min_score: float = 0.7
    skip_function_words: bool = True


# =============================================================================
# GENRE PROFILES
# =============================================================================

PROFILES = {
    Genre.CLASSIC_EAST: GenreProfile(
        genre=Genre.CLASSIC_EAST,
        name="Classic East Coast",
        description="90s NYC boom bap - Nas, Biggie, Wu-Tang",
        typical_bpm_range=(85, 100),
        preferred_grid=GridType.SIXTEENTH,
        typical_syllables_per_bar=(20, 35),   # Adjusted for full syllable count
        typical_rhyme_density=(4.0, 10.0),    # Higher when counting all rhymes
        expect_multisyllabic=True,
        expect_internal_rhymes=True,
        typical_deviation_ms=(-10, 15),
        laid_back_tendency=0.1,
        expected_melodicity=(0.0, 0.15),
        rhyme_min_score=0.75,
    ),

    Genre.CLASSIC_WEST: GenreProfile(
        genre=Genre.CLASSIC_WEST,
        name="Classic West Coast",
        description="G-funk era - Snoop, Dre, 2Pac",
        typical_bpm_range=(88, 108),
        preferred_grid=GridType.SIXTEENTH,
        typical_syllables_per_bar=(18, 32),   # Adjusted
        typical_rhyme_density=(3.0, 8.0),     # Adjusted
        expect_multisyllabic=False,           # Less complex rhymes
        typical_deviation_ms=(-5, 25),
        laid_back_tendency=0.3,               # Very laid back
        expected_melodicity=(0.1, 0.3),
        allows_singing=True,                  # Hooks often sung
        rhyme_min_score=0.7,
    ),

    Genre.CLASSIC_SOUTH: GenreProfile(
        genre=Genre.CLASSIC_SOUTH,
        name="Classic Southern",
        description="Early crunk, Three 6, Outkast",
        typical_bpm_range=(70, 90),
        preferred_grid=GridType.SIXTEENTH,
        allows_triplet_feel=True,
        typical_syllables_per_bar=(16, 28),   # Adjusted
        typical_rhyme_density=(3.0, 8.0),     # Adjusted
        typical_deviation_ms=(-15, 20),
        laid_back_tendency=0.15,
        expected_melodicity=(0.1, 0.4),
        allows_singing=True,
        rhyme_min_score=0.65,
    ),

    Genre.TRAP: GenreProfile(
        genre=Genre.TRAP,
        name="Trap",
        description="Modern Atlanta - Future, Migos, Young Thug",
        typical_bpm_range=(130, 170),       # Double time feel
        preferred_grid=GridType.SIXTEENTH,
        allows_triplet_feel=True,           # Triplet flow common
        typical_swing=0.55,
        typical_syllables_per_bar=(6, 12),  # Often sparser
        typical_rhyme_density=(0.5, 2.0),
        expect_multisyllabic=False,
        expect_internal_rhymes=False,
        typical_deviation_ms=(-30, 30),
        laid_back_tendency=0.0,             # Variable
        expected_melodicity=(0.2, 0.6),
        allows_singing=True,
        rhyme_min_score=0.6,
        skip_function_words=True,
    ),

    Genre.DRILL: GenreProfile(
        genre=Genre.DRILL,
        name="Drill",
        description="UK/Chicago drill - Pop Smoke, Chief Keef",
        typical_bpm_range=(138, 145),
        preferred_grid=GridType.SIXTEENTH,
        allows_triplet_feel=True,
        typical_syllables_per_bar=(6, 14),
        typical_rhyme_density=(0.8, 2.5),
        typical_deviation_ms=(-20, 25),
        laid_back_tendency=0.1,
        expected_melodicity=(0.15, 0.45),
        allows_singing=True,
        rhyme_min_score=0.65,
    ),

    Genre.SOUNDCLOUD: GenreProfile(
        genre=Genre.SOUNDCLOUD,
        name="SoundCloud Rap",
        description="Lil Peep, XXXTentacion, Juice WRLD",
        typical_bpm_range=(140, 180),
        preferred_grid=GridType.SIXTEENTH,
        typical_syllables_per_bar=(4, 10),   # Often sparse
        typical_rhyme_density=(0.5, 1.5),
        expect_multisyllabic=False,
        expect_internal_rhymes=False,
        end_rhyme_importance=0.9,
        typical_deviation_ms=(-40, 40),      # Loose timing
        laid_back_tendency=0.0,
        expected_melodicity=(0.4, 0.9),       # Very melodic
        allows_singing=True,
        rhyme_min_score=0.55,                 # Looser rhyming
    ),

    Genre.LYRICAL: GenreProfile(
        genre=Genre.LYRICAL,
        name="Lyrical/Conscious",
        description="Kendrick, J Cole, Joey Badass",
        typical_bpm_range=(75, 95),           # Tighter BPM range
        preferred_grid=GridType.SIXTEENTH,
        typical_syllables_per_bar=(28, 45),   # Very dense, adjusted
        typical_rhyme_density=(6.0, 12.0),    # High rhyme density
        expect_multisyllabic=True,
        expect_internal_rhymes=True,
        end_rhyme_importance=0.7,
        typical_deviation_ms=(-15, 15),
        laid_back_tendency=0.05,
        expected_melodicity=(0.05, 0.3),
        allows_singing=True,
        rhyme_min_score=0.75,
    ),

    Genre.HORRORCORE: GenreProfile(
        genre=Genre.HORRORCORE,
        name="Horrorcore",
        description="ICP, Twiztid, Brotha Lynch",
        typical_bpm_range=(80, 110),
        preferred_grid=GridType.SIXTEENTH,
        typical_syllables_per_bar=(10, 18),
        typical_rhyme_density=(1.5, 3.5),
        expect_multisyllabic=True,
        typical_deviation_ms=(-10, 20),
        laid_back_tendency=0.1,
        expected_melodicity=(0.1, 0.4),
        allows_singing=True,
        rhyme_min_score=0.7,
    ),

    Genre.MELODIC: GenreProfile(
        genre=Genre.MELODIC,
        name="Melodic Rap",
        description="Drake, Post Malone, modern hooks",
        typical_bpm_range=(60, 90),
        preferred_grid=GridType.SIXTEENTH,
        typical_syllables_per_bar=(6, 12),
        typical_rhyme_density=(0.5, 2.0),
        expect_multisyllabic=False,
        end_rhyme_importance=0.9,
        typical_deviation_ms=(-25, 25),
        laid_back_tendency=0.2,
        expected_melodicity=(0.5, 1.0),       # Highly melodic
        allows_singing=True,
        rhyme_min_score=0.6,
    ),
}


def get_profile(genre: Genre) -> GenreProfile:
    """Get profile for a genre"""
    return PROFILES.get(genre, PROFILES[Genre.CLASSIC_EAST])


def detect_genre_from_features(
    bpm: float,
    syllables_per_bar: float,
    melodicity: float,
    rhyme_density: float
) -> tuple:
    """
    Guess genre from audio features.
    Returns (Genre, confidence)
    """
    scores = {}

    for genre, profile in PROFILES.items():
        score = 0.0
        count = 0

        # BPM match
        low, high = profile.typical_bpm_range
        if low <= bpm <= high:
            score += 1.0
        elif low - 20 <= bpm <= high + 20:
            score += 0.5
        count += 1

        # Syllable density match
        low, high = profile.typical_syllables_per_bar
        if low <= syllables_per_bar <= high:
            score += 1.0
        elif low - 4 <= syllables_per_bar <= high + 4:
            score += 0.5
        count += 1

        # Melodicity match
        low, high = profile.expected_melodicity
        if low <= melodicity <= high:
            score += 1.0
        elif low - 0.2 <= melodicity <= high + 0.2:
            score += 0.5
        count += 1

        # Rhyme density match
        low, high = profile.typical_rhyme_density
        if low <= rhyme_density <= high:
            score += 1.0
        count += 1

        scores[genre] = score / count

    best_genre = max(scores, key=scores.get)
    return best_genre, scores[best_genre]


def adjust_for_genre(
    base_config: dict,
    genre: Genre
) -> dict:
    """
    Adjust analysis configuration based on genre.
    Returns modified config dict.
    """
    profile = get_profile(genre)
    config = base_config.copy()

    # Adjust rhyme detection
    config['rhyme_min_score'] = profile.rhyme_min_score
    config['skip_function_words'] = profile.skip_function_words
    config['expect_multisyllabic'] = profile.expect_multisyllabic

    # Adjust microtiming interpretation
    config['laid_back_threshold'] = profile.laid_back_tendency + 10
    config['rushing_threshold'] = profile.laid_back_tendency - 10

    # Adjust melodicity expectations
    config['melodicity_threshold'] = profile.expected_melodicity[1]

    return config


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def describe_profile(genre: Genre) -> str:
    """Human-readable profile description"""
    p = get_profile(genre)
    return f"""
Genre: {p.name}
Description: {p.description}

RHYTHM:
  BPM range: {p.typical_bpm_range[0]}-{p.typical_bpm_range[1]}
  Grid: {p.preferred_grid.value}
  Triplet feel: {'Yes' if p.allows_triplet_feel else 'No'}
  Swing: {p.typical_swing:.2f}

DENSITY:
  Syllables/bar: {p.typical_syllables_per_bar[0]}-{p.typical_syllables_per_bar[1]}
  Rhymes/bar: {p.typical_rhyme_density[0]:.1f}-{p.typical_rhyme_density[1]:.1f}

STYLE:
  Melodicity: {p.expected_melodicity[0]:.1f}-{p.expected_melodicity[1]:.1f}
  Laid-back tendency: {'+' if p.laid_back_tendency > 0 else ''}{p.laid_back_tendency * 10:.0f}ms
  Allows singing: {'Yes' if p.allows_singing else 'No'}

RHYME CONFIG:
  Min score: {p.rhyme_min_score}
  Expect multisyllabic: {'Yes' if p.expect_multisyllabic else 'No'}
  Expect internal: {'Yes' if p.expect_internal_rhymes else 'No'}
"""


if __name__ == "__main__":
    from models import Genre

    print("=" * 60)
    print("GENRE PROFILES")
    print("=" * 60)

    for genre in Genre:
        print(describe_profile(genre))
        print("-" * 60)
