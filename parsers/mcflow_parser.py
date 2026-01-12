"""
mcflow_parser.py - Parser for MCFlow .rap files (Humdrum format)

Humdrum spines:
    **recip   - Rhythmic duration
    **stress  - Stress level (0, 1, or 2)
    **tone    - Pitch contour
    **break   - Prosodic break level (1-5)
    **rhyme   - Rhyme class (A-Z)
    **ipa     - IPA transcription
    **lyrics  - English text
    **hype    - Secondary vocals
"""

import re
from pathlib import Path
from typing import Optional, Iterator
from .ground_truth import GroundTruthSyllable, GroundTruthVerse


class MCFlowParser:
    """Parser for MCFlow .rap files (Humdrum format)."""

    SPINE_NAMES = ['recip', 'stress', 'tone', 'break', 'rhyme', 'ipa', 'lyrics', 'hype']

    def __init__(self, mcflow_dir: Optional[str] = None):
        self.mcflow_dir = Path(mcflow_dir) if mcflow_dir else None

    def parse_file(self, filepath: str) -> GroundTruthVerse:
        """Parse a single .rap file"""
        filepath = Path(filepath)

        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        return self._parse_lines(lines, filepath)

    def parse_all(self, directory: Optional[str] = None) -> Iterator[GroundTruthVerse]:
        """Parse all .rap files in directory"""
        dir_path = Path(directory) if directory else self.mcflow_dir
        if not dir_path:
            raise ValueError("No directory specified")

        for rap_file in sorted(dir_path.glob("*.rap")):
            try:
                yield self.parse_file(rap_file)
            except Exception as e:
                print(f"Warning: Failed to parse {rap_file.name}: {e}")

    def _parse_lines(self, lines: list, filepath: Path) -> GroundTruthVerse:
        """Parse Humdrum content"""
        filename = filepath.stem
        parts = filename.split('_', 1)
        artist = parts[0] if parts else "Unknown"
        title = parts[1] if len(parts) > 1 else filename

        spine_order = []
        syllables = []
        current_bar = 1
        current_beat = 1.0
        bpm = None
        time_sig = (4, 4)
        syl_idx = 0

        for line in lines:
            line = line.strip()

            if not line or line.startswith('!'):
                continue

            if line.startswith('**'):
                spine_order = self._parse_spine_header(line)
                continue

            if line.startswith('*'):
                meta = self._parse_interpretation(line)
                if 'bpm' in meta:
                    bpm = meta['bpm']
                if 'time_sig' in meta:
                    time_sig = meta['time_sig']
                if 'bar' in meta:
                    current_bar = meta['bar']
                    current_beat = 1.0
                continue

            if line.startswith('='):
                bar_num = self._parse_barline(line)
                if bar_num:
                    current_bar = bar_num
                else:
                    current_bar += 1
                current_beat = 1.0
                continue

            if '\t' in line and spine_order:
                syl = self._parse_data_record(
                    line, spine_order, current_bar, current_beat, syl_idx
                )
                if syl:
                    syllables.append(syl)
                    current_beat += syl.duration_beats
                    syl_idx += 1

        return GroundTruthVerse(
            id=f"mcflow_{filename}",
            artist=self._clean_artist_name(artist),
            title=self._clean_title(title),
            syllables=syllables,
            bpm=bpm,
            time_signature=time_sig,
            source="mcflow",
            source_file=str(filepath)
        )

    def _parse_spine_header(self, line: str) -> list:
        spines = line.split('\t')
        order = []
        for spine in spines:
            spine_name = spine.lstrip('*').lower()
            if spine_name in self.SPINE_NAMES:
                order.append(spine_name)
            else:
                order.append(None)
        return order

    def _parse_interpretation(self, line: str) -> dict:
        meta = {}
        parts = line.split('\t')

        for part in parts:
            part = part.strip('*')
            if part.startswith('MM'):
                try:
                    meta['bpm'] = float(part[2:])
                except ValueError:
                    pass
            if part.startswith('M') and '/' in part:
                try:
                    num, denom = part[1:].split('/')
                    meta['time_sig'] = (int(num), int(denom))
                except ValueError:
                    pass

        return meta

    def _parse_barline(self, line: str) -> Optional[int]:
        match = re.search(r'=(\d+)', line)
        if match:
            return int(match.group(1))
        return None

    def _parse_data_record(
        self, line: str, spine_order: list, bar: int, beat: float, syl_idx: int
    ) -> Optional[GroundTruthSyllable]:
        fields = line.split('\t')
        data = {}

        for i, spine_name in enumerate(spine_order):
            if spine_name and i < len(fields):
                value = fields[i].strip()
                if value and value != '.':
                    data[spine_name] = value

        if 'lyrics' not in data:
            return None

        text = data['lyrics']
        duration = self._parse_recip(data.get('recip', '16'))
        stress_val = data.get('stress', '0')
        stressed = stress_val in ('1', '2')

        break_level = 0
        if 'break' in data:
            try:
                break_level = int(data['break'])
            except ValueError:
                pass

        rhyme_class = data.get('rhyme')
        if rhyme_class == '.':
            rhyme_class = None

        phonemes = []
        if 'ipa' in data:
            ipa = data['ipa']
            phonemes = self._ipa_to_arpabet(ipa)

        return GroundTruthSyllable(
            id=f"mcflow_syl_{syl_idx}",
            text=text,
            word=text,
            bar=bar,
            beat=beat,
            duration_beats=duration,
            quantization=16,
            phonemes=phonemes,
            stressed=stressed,
            break_level=break_level,
            line_ending=break_level >= 3,
            breath_ending=break_level >= 4,
            rhyme_class=rhyme_class,
            source="mcflow"
        )

    def _parse_recip(self, recip: str) -> float:
        if not recip or recip == '.':
            return 0.25

        if '%' in recip:
            try:
                num, denom = recip.split('%')
                return float(num) / float(denom)
            except ValueError:
                pass

        dots = recip.count('.')
        base = recip.rstrip('.')

        try:
            duration = 4.0 / float(base)
            dot_add = duration
            for _ in range(dots):
                dot_add /= 2
                duration += dot_add
            return duration
        except ValueError:
            return 0.25

    def _ipa_to_arpabet(self, ipa: str) -> list:
        return [p for p in re.split(r'[\s\.]', ipa) if p]

    def _clean_artist_name(self, name: str) -> str:
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', name)

    def _clean_title(self, title: str) -> str:
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', title)
