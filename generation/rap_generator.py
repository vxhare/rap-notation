"""
Rap Lyrics Generator

Generates rap bars based on analysis findings:
- Matches syllable density per bar
- Uses detected rhyme schemes (AABB, ABAB, etc.)
- Incorporates adlibs at detected frequency
- Doubles emphasized words
- Fits noctuplet or 16th note grid

Usage:
    from generation.rap_generator import RapGenerator

    generator = RapGenerator()
    song = generator.generate_from_analysis(analysis_data)
    print(song.to_text())
"""

import random
import pronouncing
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# Word banks by category
WORD_BANKS = {
    "flex": [
        "money", "cash", "bands", "racks", "stacks", "paper", "bread", "cheese",
        "drip", "ice", "chains", "watch", "whip", "foreign", "lambo", "rarri",
        "boss", "king", "legend", "goat", "greatest", "winning", "champion"
    ],
    "motion": [
        "go", "run", "move", "slide", "glide", "float", "fly", "soar",
        "push", "pull", "grind", "hustle", "work", "build", "rise", "climb"
    ],
    "emotion": [
        "feel", "love", "hate", "want", "need", "miss", "wish", "hope",
        "dream", "think", "know", "believe", "trust", "doubt", "fear"
    ],
    "people": [
        "crew", "squad", "team", "gang", "fam", "homies", "brothers",
        "haters", "opps", "snakes", "fakes", "real ones", "day ones"
    ],
    "time": [
        "now", "then", "always", "never", "forever", "tonight", "today",
        "tomorrow", "lately", "lately", "sometimes", "everyday"
    ],
    "place": [
        "city", "block", "hood", "streets", "top", "bottom", "throne",
        "game", "world", "stage", "spotlight", "shadows"
    ],
    "action": [
        "hit", "drop", "pop", "lock", "rock", "knock", "stop", "chop",
        "flip", "dip", "trip", "skip", "rip", "grip", "ship", "sip"
    ],
    "adlibs": [
        "what", "yeah", "uh", "ay", "skrt", "brr", "gang", "woo",
        "let's go", "facts", "real talk", "no cap", "sheesh", "damn"
    ]
}

# Sentence templates with placeholders
TEMPLATES = {
    "flex": [
        "I got {flex} on my {flex}",
        "They know I'm the {flex}",
        "Count up the {flex}, yeah",
        "{flex} {action}, never stop",
        "Real {flex} recognize {flex}",
        "All this {flex}, can't deny",
    ],
    "motion": [
        "Let's {motion}, let's {motion}",
        "Watch me {motion} on these {people}",
        "I {motion} different, they don't understand",
        "Had to {motion} up, no looking back",
        "We gon' {motion} 'til we reach the {place}",
    ],
    "emotion": [
        "I {emotion} like I'm the only one",
        "Don't {emotion} nothing for these {people}",
        "They don't {emotion} what I'm going through",
        "I {emotion} it when they {emotion} it",
        "Can't {emotion} what you never had",
    ],
    "story": [
        "Started from the {place}, now I'm at the {place}",
        "Back in the {time}, I was on the {place}",
        "Every {time}, I think about the {place}",
        "{time} I was broke, {time} I'm not",
        "From the {place} to the {place}, that's the way",
    ],
    "bars": [
        "{action} the beat, then I {action} the track",
        "I {action} it down, bring it right back",
        "Watch me {action}, watch me {action}",
        "Had to {action} twice, make 'em look back",
    ]
}


class RhymeScheme(Enum):
    AABB = "aabb"  # Couplets
    ABAB = "abab"  # Alternate
    ABBA = "abba"  # Enclosed
    AAAA = "aaaa"  # Monorhyme
    FREE = "free"  # Mixed


@dataclass
class GeneratedBar:
    """A single bar of generated lyrics."""
    text: str
    syllable_count: int
    end_word: str
    rhyme_sound: str
    has_adlib: bool
    adlib_text: Optional[str] = None
    doubled_words: list = field(default_factory=list)
    beat_positions: list = field(default_factory=list)  # Which beats have syllables


@dataclass
class GeneratedVerse:
    """A verse of generated bars."""
    bars: list[GeneratedBar]
    rhyme_scheme: RhymeScheme


@dataclass
class GeneratedSong:
    """Complete generated song."""
    title: str
    bpm: int
    verses: list[GeneratedVerse]
    hooks: list[GeneratedVerse]
    style: str

    def to_text(self) -> str:
        """Convert to readable text format."""
        lines = []
        lines.append(f"# {self.title}")
        lines.append(f"# BPM: {self.bpm} | Style: {self.style}")
        lines.append("")

        section_num = 1
        for i, verse in enumerate(self.verses):
            lines.append(f"[Verse {i+1}]")
            for bar in verse.bars:
                text = bar.text
                if bar.has_adlib and bar.adlib_text:
                    text += f" *({bar.adlib_text})*"
                if bar.doubled_words:
                    # Mark doubled words
                    for dw in bar.doubled_words:
                        text = text.replace(dw, f"[{dw}]", 1)
                lines.append(text)
            lines.append("")

            # Add hook after verse if available
            if i < len(self.hooks):
                lines.append("[Hook]")
                for bar in self.hooks[i].bars:
                    text = bar.text
                    if bar.has_adlib and bar.adlib_text:
                        text += f" *({bar.adlib_text})*"
                    lines.append(text)
                lines.append("")

        return "\n".join(lines)

    def to_flow_data(self) -> dict:
        """Convert to flow diagram compatible format."""
        syllables = []
        bar_num = 1

        for verse in self.verses:
            for bar in verse.bars:
                # Simple syllable placement (would need proper alignment)
                words = bar.text.split()
                beat = 1.0
                beat_increment = 4.0 / max(len(words), 1)

                for word in words:
                    syllables.append({
                        "text": word,
                        "bar": bar_num,
                        "beat": beat,
                        "is_doubled": word in bar.doubled_words
                    })
                    beat += beat_increment

                bar_num += 1

        return {
            "title": self.title,
            "bpm": self.bpm,
            "syllables": syllables,
            "total_bars": bar_num - 1
        }


class RapGenerator:
    """
    Generate rap lyrics based on analysis patterns.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)

        self.rhyme_cache = {}

    def generate_from_analysis(
        self,
        analysis: dict,
        num_verses: int = 2,
        bars_per_verse: int = 8,
        include_hook: bool = True,
        topic: str = "flex"
    ) -> GeneratedSong:
        """
        Generate a song based on analysis findings.

        Args:
            analysis: Analysis dict from pipeline
            num_verses: Number of verses to generate
            bars_per_verse: Bars per verse
            include_hook: Whether to generate hooks
            topic: Theme ("flex", "motion", "emotion", "story")

        Returns:
            GeneratedSong object
        """
        # Extract parameters from analysis
        bpm = int(analysis.get("beat", {}).get("bpm", 120))

        # Target syllables per bar from analysis
        total_syls = analysis.get("transcription", {}).get("syllable_count", 400)
        total_bars = analysis.get("beat", {}).get("total_bars", 100)
        target_density = total_syls / total_bars if total_bars > 0 else 5

        # Rhyme density
        rhyme_density = analysis.get("rhyme_scheme", {}).get("density", 1.0)

        # Adlib frequency
        adlib_info = analysis.get("adlibs", {})
        adlibs_per_bar = adlib_info.get("per_bar", 0.1)
        common_adlibs = [a[0] for a in adlib_info.get("most_common", [("what", 1)])]

        # Most doubled words pattern
        doubled_info = analysis.get("vocal_doubles", {})
        doubled_pct = doubled_info.get("doubled_percentage", 10) / 100

        # Determine rhyme scheme from pattern
        pattern = analysis.get("rhyme_scheme", {}).get("pattern", "AABB")
        if "AABB" in pattern[:8]:
            rhyme_scheme = RhymeScheme.AABB
        elif "ABAB" in pattern[:8]:
            rhyme_scheme = RhymeScheme.ABAB
        else:
            rhyme_scheme = RhymeScheme.AABB

        # Genre/style
        style = analysis.get("genre", {}).get("detected", "trap")

        # Generate verses
        verses = []
        hooks = []

        for v in range(num_verses):
            verse = self._generate_verse(
                bars=bars_per_verse,
                target_density=target_density,
                rhyme_scheme=rhyme_scheme,
                adlib_frequency=adlibs_per_bar,
                adlib_options=common_adlibs if common_adlibs else ["what", "yeah"],
                double_frequency=doubled_pct,
                topic=topic
            )
            verses.append(verse)

            if include_hook:
                hook = self._generate_verse(
                    bars=4,
                    target_density=target_density * 0.8,  # Hooks slightly less dense
                    rhyme_scheme=RhymeScheme.AABB,
                    adlib_frequency=adlibs_per_bar * 1.5,  # More adlibs in hooks
                    adlib_options=common_adlibs if common_adlibs else ["what", "yeah"],
                    double_frequency=doubled_pct * 1.5,  # More doubles in hooks
                    topic=topic
                )
                hooks.append(hook)

        # Generate title
        title = self._generate_title(topic, style)

        return GeneratedSong(
            title=title,
            bpm=bpm,
            verses=verses,
            hooks=hooks,
            style=style
        )

    def _generate_verse(
        self,
        bars: int,
        target_density: float,
        rhyme_scheme: RhymeScheme,
        adlib_frequency: float,
        adlib_options: list,
        double_frequency: float,
        topic: str
    ) -> GeneratedVerse:
        """Generate a single verse."""
        generated_bars = []

        # Group bars by rhyme scheme
        if rhyme_scheme == RhymeScheme.AABB:
            groups = [(i, i+1) for i in range(0, bars, 2)]
        elif rhyme_scheme == RhymeScheme.ABAB:
            groups = [(i, i+2) for i in range(0, bars-2, 2)]
        else:
            groups = [(i,) for i in range(bars)]

        # Generate rhyming pairs
        used_rhymes = set()
        bar_index = 0

        while bar_index < bars:
            # Pick a seed word for this rhyme group
            seed_word = self._pick_word(topic)
            rhymes = self._get_rhymes(seed_word)

            if not rhymes:
                seed_word = random.choice(WORD_BANKS["action"])
                rhymes = self._get_rhymes(seed_word)

            if not rhymes:
                rhymes = [seed_word]

            # Generate 2 bars that rhyme (for AABB)
            for i in range(min(2, bars - bar_index)):
                end_word = rhymes[i % len(rhymes)] if i > 0 else seed_word

                bar = self._generate_bar(
                    target_syllables=int(target_density),
                    end_word=end_word,
                    topic=topic,
                    add_adlib=random.random() < adlib_frequency,
                    adlib_options=adlib_options,
                    double_word=random.random() < double_frequency
                )
                generated_bars.append(bar)
                bar_index += 1

        return GeneratedVerse(
            bars=generated_bars,
            rhyme_scheme=rhyme_scheme
        )

    def _generate_bar(
        self,
        target_syllables: int,
        end_word: str,
        topic: str,
        add_adlib: bool,
        adlib_options: list,
        double_word: bool
    ) -> GeneratedBar:
        """Generate a single bar ending with the given word."""

        # Pick a template
        templates = TEMPLATES.get(topic, TEMPLATES["bars"])
        template = random.choice(templates)

        # Fill template
        text = template
        for category in ["flex", "motion", "emotion", "people", "time", "place", "action"]:
            while f"{{{category}}}" in text:
                word = random.choice(WORD_BANKS.get(category, ["thing"]))
                text = text.replace(f"{{{category}}}", word, 1)

        # Ensure it ends with rhyme word (append or replace last word)
        words = text.split()
        if words:
            words[-1] = end_word
        else:
            words = [end_word]

        text = " ".join(words)

        # Count syllables
        syllable_count = self._count_syllables(text)

        # Adjust to target (simple padding/trimming)
        while syllable_count < target_syllables - 2:
            filler = random.choice(["yeah", "I", "got", "the", "my", "with"])
            words.insert(random.randint(0, len(words)-1), filler)
            text = " ".join(words)
            syllable_count = self._count_syllables(text)

        # Handle adlib
        adlib_text = None
        if add_adlib:
            adlib_text = random.choice(adlib_options)

        # Handle doubled word
        doubled_words = []
        if double_word and words:
            # Double an emphasis word (often verbs or the first word)
            double_candidates = [w for w in words if len(w) > 2]
            if double_candidates:
                doubled_words.append(random.choice(double_candidates))

        # Get rhyme sound
        phones = pronouncing.phones_for_word(end_word)
        rhyme_sound = pronouncing.rhyming_part(phones[0]) if phones else end_word[-2:]

        return GeneratedBar(
            text=text.capitalize(),
            syllable_count=syllable_count,
            end_word=end_word,
            rhyme_sound=rhyme_sound,
            has_adlib=add_adlib,
            adlib_text=adlib_text,
            doubled_words=doubled_words
        )

    def _get_rhymes(self, word: str) -> list:
        """Get rhyming words."""
        if word in self.rhyme_cache:
            return self.rhyme_cache[word]

        rhymes = pronouncing.rhymes(word)
        # Filter to common words (shorter = more common usually)
        rhymes = [r for r in rhymes if len(r) <= 8][:20]

        self.rhyme_cache[word] = rhymes
        return rhymes

    def _pick_word(self, topic: str) -> str:
        """Pick a word from topic category."""
        bank = WORD_BANKS.get(topic, WORD_BANKS["flex"])
        return random.choice(bank)

    def _count_syllables(self, text: str) -> int:
        """Count syllables in text."""
        count = 0
        for word in text.split():
            word_clean = word.strip(".,!?'\"")
            phones = pronouncing.phones_for_word(word_clean.lower())
            if phones:
                count += pronouncing.syllable_count(phones[0])
            else:
                # Estimate: count vowel groups
                vowels = "aeiouAEIOU"
                count += max(1, sum(1 for i, c in enumerate(word_clean)
                                   if c in vowels and (i == 0 or word_clean[i-1] not in vowels)))
        return count

    def _generate_title(self, topic: str, style: str) -> str:
        """Generate a song title."""
        title_templates = [
            "{word} Mode",
            "{word} Flow",
            "No {word}",
            "{word} Talk",
            "Real {word}",
            "{word} Season",
            "All {word}",
            "{word} Mentality"
        ]

        word = random.choice(WORD_BANKS.get(topic, WORD_BANKS["flex"]))
        template = random.choice(title_templates)
        return template.format(word=word.capitalize())


def generate_from_file(analysis_path: str, topic: str = "flex") -> str:
    """
    Convenience function to generate from analysis JSON file.

    Args:
        analysis_path: Path to analysis JSON
        topic: Theme for lyrics

    Returns:
        Generated lyrics as string
    """
    import json

    with open(analysis_path) as f:
        analysis = json.load(f)

    generator = RapGenerator()
    song = generator.generate_from_analysis(analysis, topic=topic)

    return song.to_text()


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python rap_generator.py <analysis.json> [topic]")
        print("Topics: flex, motion, emotion, story, bars")
        sys.exit(1)

    analysis_path = sys.argv[1]
    topic = sys.argv[2] if len(sys.argv) > 2 else "flex"

    with open(analysis_path) as f:
        analysis = json.load(f)

    generator = RapGenerator(seed=42)  # Reproducible
    song = generator.generate_from_analysis(
        analysis,
        num_verses=2,
        bars_per_verse=8,
        topic=topic
    )

    print(song.to_text())

    # Also save flow data
    flow_data = song.to_flow_data()
    output_path = analysis_path.replace(".json", "_generated.json")
    with open(output_path, "w") as f:
        json.dump(flow_data, f, indent=2)
    print(f"\nFlow data saved to: {output_path}")
