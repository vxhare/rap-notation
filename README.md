# Rap Notation Analysis System

A comprehensive system for analyzing rap performances into structured, multi-layer notation data. Based on Martin Connor's thesis work on rap flow notation systems.

## What This System Does

Takes a rap audio file and produces:
- **Flow diagrams** (Adams-style 16th note / noctuplet grids)
- **Rhyme scheme analysis** with color-coded visualization
- **Microtiming profiles** (how the rapper relates to the beat)
- **Melodicity scoring** (rap vs. singing detection)
- **Vocal double detection** (stacked vocals identification)
- **Production analysis** (reverb, adlibs, beat vocals)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python pipeline.py path/to/track.mp3 "Artist Name"

# Output: track_analysis.json + flow diagram HTML
```

## Accomplishments & Technical Innovations

### 1. Multi-Layer Syllable Alignment

**Problem**: Whisper gives word-level timestamps, but rap analysis needs syllable-level precision.

**Our Solution**:
- Aggressive onset detection tuned for vocal transients
- Greedy alignment algorithm that splits words into syllables at detected onsets
- Syllable count estimation using pyphen + CMU dictionary fallback
- Result: Sub-beat accuracy for syllable placement

**Improvement over existing tools**: Most tools either use word-level only (too coarse) or require forced alignment models like MFA (slow, requires training data). We achieve syllable-level without external alignment models.

---

### 2. Phoneme-Based Rhyme Detection

**Problem**: Simple string matching misses slant rhymes, assonance, and multisyllabic rhymes that are essential to rap.

**Our Solution**:
- CMU Pronouncing Dictionary lookup with G2P fallback for unknown words
- Rime extraction (nucleus + coda) for rhyme comparison
- Multiple rhyme type detection:
  - Perfect rhymes (cat/hat)
  - Slant rhymes (cat/bed)
  - Assonance (vowel matching)
  - Consonance (consonant matching)
  - Multisyllabic (fantastic/gymnastic)
- Weighted scoring based on phonetic similarity

**Improvement over existing tools**: Most rhyme detectors only find perfect end rhymes. We detect internal rhymes, slant rhymes, and complex multi-syllabic patterns that define sophisticated rap flows.

---

### 3. Dual Grid Alignment (16th Note + Noctuplet)

**Problem**: Rap doesn't fit neatly into standard 16th note grids. Many flows use triplet feels.

**Our Solution**:
- Simultaneous alignment to both 16th note (4 per beat) and noctuplet (4.5 per beat) grids
- Deviation calculation for each grid
- Automatic recommendation of best-fit grid
- Quantization error tracking for microtiming analysis

**Improvement over existing tools**: Based on Martin Connor's research showing noctuplet grids better capture certain rap flows. Most tools force binary subdivisions.

---

### 4. Vocal Double Detection (Mid-Side Correlation)

**Problem**: Detecting when a rapper stacks/doubles their vocals vs. when vocal samples appear in the beat.

**Our Solution**:
- Mid-Side decomposition: Mid = (L+R)/2, Side = (L-R)/2
- Envelope correlation analysis between Mid and Side channels
- Key insight: True doubles have high correlation (side "shadows" mid)
- Beat vocals have low/negative correlation (independent rhythm)
- Instrumental bleed detection to filter Demucs artifacts
- Word-level detection using Whisper timestamps

**This is novel**: No existing tool specifically detects vocal doubles in this way. We developed the Mid-Side correlation approach to distinguish stacked rapper vocals from beat vocal samples.

---

### 5. Beat Vocal Filtering

**Problem**: Modern rap uses vocal samples in beats (e.g., chopped soul vocals). These get confused with the rapper's voice.

**Our Solution**:
- Stem separation via Demucs
- Pitch matching between detected "doubles" and main vocal baseline
- Spectral similarity comparison
- Mid-Side correlation to identify independent rhythmic patterns
- Result: 280+ beat vocal segments correctly filtered in test track

**This is novel**: Existing stem separation tools extract vocals but don't distinguish rapper from beat vocals. We added classification logic.

---

### 6. Melodicity Scoring

**Problem**: Modern rap exists on a spectrum from pure speech to full singing. Quantifying this is valuable.

**Our Solution**:
- Pitch tracking using PYIN algorithm
- Per-syllable melodicity scoring based on:
  - Pitch stability (sung = stable, rapped = variable)
  - Vibrato detection
  - Pitch range utilization
- Classification: sung vs. rapped per syllable
- Overall melodicity score (0 = pure rap, 1 = pure singing)

**Improvement over existing tools**: Most tools classify entire tracks. We provide per-syllable melodicity, allowing analysis of style shifts within a verse.

---

### 7. Reverb-Aware Processing

**Problem**: Heavy reverb obscures vocal onsets, leading to poor syllable detection.

**Our Solution**:
- Reverb amount estimation via:
  - Transient sharpness analysis
  - Spectral decay measurement
  - Autocorrelation for RT60 estimation
- Adaptive onset detection thresholds based on reverb profile
- Style classification (dry, room, hall, plate, cloud)

**Improvement over existing tools**: Standard onset detection fails on heavily reverbed vocals. We adapt parameters dynamically.

---

### 8. Adlib Detection

**Problem**: Adlibs ("yeah", "what", "skrrt") should be analyzed separately from main lyrics.

**Our Solution**:
- Vocabulary-based detection (common adlib words)
- Stereo panning analysis (adlibs often panned to sides)
- Confidence scoring
- Separation into main vocals vs. adlib track

**Improvement over existing tools**: Most transcription tools include adlibs inline. We separate them for cleaner analysis.

---

### 9. Interactive Flow Diagram Visualization

**Problem**: Academic flow diagrams are static and hard to read.

**Our Solution**:
- HTML/CSS visualization with:
  - Color-coded rhyme groups
  - 16th note grid layout
  - Doubled word highlighting (gold border)
  - Metric cards with analysis summary
  - Dark theme optimized for readability
- Full-track rendering with section navigation

**Improvement over existing tools**: Martin Connor's diagrams are hand-drawn. We automate the entire visualization.

---

### 10. Genre/Style Detection

**Problem**: Different rap subgenres have distinct flow characteristics.

**Our Solution**:
- Feature-based classification using:
  - BPM ranges
  - Syllable density
  - Melodicity scores
  - Rhyme density
- Supported styles: Boom bap, Trap, Drill, Melodic rap, Conscious, Mumble
- Confidence scoring

---

## Architecture

```
audio.mp3
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  PIPELINE                                                    │
│                                                              │
│  1. Beat Detection (librosa)                                │
│  2. Reverb Analysis → Adaptive onset params                 │
│  3. Whisper Transcription → Word timestamps                 │
│  4. Adlib Detection → Separate main/adlib                   │
│  5. Onset Detection → Syllable boundaries                   │
│  6. Phoneme Lookup → CMU dict + G2P                         │
│  7. Rhyme Detection → Multi-type matching                   │
│  8. Grid Alignment → 16th + noctuplet                       │
│  9. Pitch/Melodicity → Per-syllable scoring                 │
│ 10. Vocal Doubles → Mid-Side correlation                    │
│ 11. Genre Detection → Feature classification                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
analysis.json + flow_diagram.html
```

## File Structure

```
rap-notation/
├── pipeline.py              # Main orchestration
├── models.py                # Data models (dataclasses)
├── requirements.txt         # Dependencies
│
├── audio/
│   ├── onset_detection.py   # Vocal onset detection
│   ├── syllable_alignment.py # Word→syllable splitting
│   ├── pitch_tracking.py    # PYIN + melodicity
│   ├── vocal_doubles.py     # Mid-Side double detection
│   ├── reverb_detection.py  # Reverb estimation
│   └── adlib_detection.py   # Adlib identification
│
├── phonetics/
│   └── phoneme_lookup.py    # CMU dict + G2P
│
├── analysis/
│   ├── rhyme_detector.py    # Multi-type rhyme detection
│   ├── grid_alignment.py    # Dual grid quantization
│   └── flow_metrics.py      # Density, complexity, etc.
│
├── config/
│   └── genre_profiles.py    # Genre classification
│
├── parsers/
│   ├── mcflow_parser.py     # MCFlow dataset parser
│   └── flowbook_parser.py   # FlowBook dataset parser
│
└── tests/
    └── *.py                 # Test suites
```

## Output Example

```json
{
  "metadata": {
    "title": "Track Name",
    "artist": "Artist",
    "duration_seconds": 194.4
  },
  "beat": {
    "bpm": 123.0,
    "total_bars": 99
  },
  "syllables": [
    {
      "id": "syl_0",
      "text": "I",
      "bar": 2,
      "beat": 3.5,
      "phonemes": ["AY1"],
      "melodicity": 0.15
    }
  ],
  "rhyme_scheme": {
    "groups": [...],
    "density": 1.10
  },
  "vocal_doubles": {
    "total": 41,
    "clean_doubles": 41,
    "most_doubled_words": [["go", 6], ["let's", 4]]
  },
  "melodicity": {
    "overall": 0.23,
    "style": "rap",
    "sung_ratio": 0.24
  }
}
```

## Comparison to Existing Tools

| Feature | Our System | MCFlow | FlowBook | Praat |
|---------|------------|--------|----------|-------|
| Syllable-level timing | ✅ | ✅ | ❌ | ✅ |
| Automatic (no manual annotation) | ✅ | ❌ | ❌ | ❌ |
| Multi-type rhyme detection | ✅ | ❌ | Partial | ❌ |
| Noctuplet grid support | ✅ | ❌ | ❌ | ❌ |
| Vocal double detection | ✅ | ❌ | ❌ | ❌ |
| Beat vocal filtering | ✅ | ❌ | ❌ | ❌ |
| Melodicity scoring | ✅ | ❌ | ❌ | Partial |
| Reverb-aware processing | ✅ | ❌ | ❌ | ❌ |
| HTML visualization | ✅ | ❌ | ❌ | ❌ |

## Known Limitations

We believe in transparency. Here's what doesn't work perfectly:

### Transcription (Whisper)
- **Word-level only** - we estimate syllables, not perfect
- **Struggles with**: mumble rap, heavy accents, regional slang
- **Mishears lyrics** - especially made-up words, ad-libs, fast flows
- **Merges/drops words** in very fast double-time sections

### Syllable Alignment
- Greedy onset alignment isn't always accurate
- pyphen/CMU dict don't know rap slang ("skrrt" = how many syllables?)
- Multi-syllabic words can split at wrong phoneme boundaries

### Onset Detection
- Still picks up bleed from the beat even after Demucs separation
- Heavy reverb adaptation helps but isn't perfect
- Very fast flows (Twista, Tech N9ne, choppers) can overwhelm it

### Rhyme Detection
- **CMU dictionary gaps** - missing tons of slang, made-up words, brand names
- **G2P fallback guesses wrong** on creative spelling ("tha", "dat", "4eva", "2chainz")
- **Slant rhyme thresholds are subjective** - what we call a rhyme might not match artist intent
- **Regional pronunciation not modeled** - Houston vowels ≠ NY vowels ≠ UK vowels

### Vocal Double Detection
- **Mono tracks break it entirely** - Mid-Side requires stereo
- Demucs artifacts still slip through sometimes
- Can't detect doubles panned identically to main vocal (rare but happens)
- Beat vocals with similar timbre to rapper can still confuse it

### BPM/Beat Detection
- librosa struggles with complex/irregular rhythms (jazz-rap, experimental)
- **No tempo change handling** - songs that speed up/slow down mid-track
- Half-time vs double-time feel is ambiguous without human context

### Grid Alignment
- 16th and noctuplet aren't the only feels - swing, triplets, polyrhythms exist
- Some flows **intentionally** don't fit any grid (that's the style)
- Quantization error doesn't tell you if timing is sloppy or intentional

### Melodicity Scoring
- PYIN pitch tracking fails on noisy/heavily reverbed audio
- **Autotune and vocoder throw it off completely** (T-Pain, Travis Scott, etc.)
- "Sung vs rapped" threshold is somewhat arbitrary

### Adlib Detection
- Vocabulary-based - **misses creative/new ad-libs**
- Assumes standard stereo mixing conventions (ad-libs panned wide)
- Center-panned ad-libs are invisible to the detector

### General
- **Tested on limited tracks** - parameters may not generalize to all subgenres
- No emotional/energy analysis (delivery intensity, aggression, vulnerability)
- **Doesn't know when "wrong" is intentional** - behind-the-beat swagger vs. sloppy timing
- Regional pronunciation variants not modeled (different cities pronounce words differently)

### What We're NOT Trying To Do
- Replace human analysis - this is a tool to assist, not replace ears
- Generate lyrics - analysis only (see our thoughts on why generation is problematic)
- Judge quality - we measure, we don't rate

---

## Based On

- Martin Connor's PhD thesis on rap notation systems
- Kyle Adams' flow diagram methodology
- MCFlow dataset structure
- CMU Pronouncing Dictionary

## Dependencies

- librosa (beat/onset detection)
- whisper (transcription)
- demucs (stem separation)
- pronouncing (CMU dict)
- pyphen (syllabification)
- g2p-en (grapheme-to-phoneme)
- numpy, scipy (signal processing)

## License

MIT
