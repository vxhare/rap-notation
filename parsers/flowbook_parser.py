"""
flowbook_parser.py - Parser for flowBook verse transcriptions (TSV format)

TSV columns:
    word        - The word text
    syllable    - Syllable breakdown
    quant       - Quantization (16 or 32)
    duration    - Duration in grid units
    rhymeClass  - Rhyme group number
    rhymeIndex  - Position within rhyme group
    lineEnding  - 1 if line ends here
    breathEnding - 1 if breath/pause here
"""

import re
import csv
from pathlib import Path
from typing import Optional, Iterator
from .ground_truth import GroundTruthSyllable, GroundTruthVerse


class FlowBookParser:
    """Parser for flowBook verse transcriptions (TSV format)."""

    def __init__(self, flowbook_dir: Optional[str] = None):
        self.flowbook_dir = Path(flowbook_dir) if flowbook_dir else None

    def parse_file(self, filepath: str) -> GroundTruthVerse:
        """Parse a single TSV file"""
        filepath = Path(filepath)

        filename = filepath.stem
        parts = filename.split('_', 1)
        artist = parts[0] if parts else "Unknown"
        title_with_num = parts[1] if len(parts) > 1 else filename
        title = re.sub(r'\d+$', '', title_with_num)

        syllables = []
        current_bar = 1
        current_beat = 1.0

        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for idx, row in enumerate(reader):
                syl = self._parse_row(row, idx, current_bar, current_beat)

                if syl:
                    syllables.append(syl)
                    current_beat += syl.duration_beats

                    while current_beat > 4.0:
                        current_beat -= 4.0
                        current_bar += 1

        return GroundTruthVerse(
            id=f"flowbook_{filename}",
            artist=self._format_artist(artist),
            title=self._format_title(title),
            syllables=syllables,
            source="flowbook",
            source_file=str(filepath)
        )

    def parse_all(self, directory: Optional[str] = None) -> Iterator[GroundTruthVerse]:
        """Parse all .txt files in directory"""
        dir_path = Path(directory) if directory else self.flowbook_dir
        if not dir_path:
            raise ValueError("No directory specified")

        for txt_file in sorted(dir_path.glob("*.txt")):
            if txt_file.name.startswith('.'):
                continue
            try:
                yield self.parse_file(txt_file)
            except Exception as e:
                print(f"Warning: Failed to parse {txt_file.name}: {e}")

    def _parse_row(
        self, row: dict, idx: int, bar: int, beat: float
    ) -> Optional[GroundTruthSyllable]:
        text = row.get('syllable') or row.get('word', '')
        if not text:
            return None

        word = row.get('word', text)
        quant = int(row.get('quant', 16))
        duration_units = int(row.get('duration', 1))

        if quant == 16:
            duration_beats = duration_units / 4.0
        elif quant == 32:
            duration_beats = duration_units / 8.0
        else:
            duration_beats = duration_units / 4.0

        rhyme_class = row.get('rhymeClass')
        if rhyme_class in ('0', '', None):
            rhyme_class = None
        else:
            try:
                rhyme_num = int(rhyme_class)
                if rhyme_num > 0:
                    rhyme_class = chr(ord('A') + rhyme_num - 1)
                else:
                    rhyme_class = None
            except ValueError:
                pass

        rhyme_index = None
        if row.get('rhymeIndex'):
            try:
                rhyme_index = int(row['rhymeIndex'])
            except ValueError:
                pass

        line_ending = row.get('lineEnding', '0') == '1'
        breath_ending = row.get('breathEnding', '0') == '1'

        return GroundTruthSyllable(
            id=f"flowbook_syl_{idx}",
            text=text,
            word=word,
            bar=bar,
            beat=beat,
            duration_beats=duration_beats,
            quantization=quant,
            stressed=False,
            line_ending=line_ending,
            breath_ending=breath_ending,
            rhyme_class=rhyme_class,
            rhyme_index=rhyme_index,
            source="flowbook"
        )

    def _format_artist(self, name: str) -> str:
        result = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        return result.title()

    def _format_title(self, title: str) -> str:
        result = re.sub(r'([a-z])([A-Z])', r'\1 \2', title)
        return result.title()
