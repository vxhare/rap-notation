"""Export true doubles (lyric-aligned) and beat vocal examples."""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from audio.vocal_double_detection_v2 import get_main_vocal_baseline, is_likely_beat_vocal
from scipy.ndimage import uniform_filter1d

# Load data
from pipeline import analyze_track
result = analyze_track('data/drake_what_did_i_miss.mp3', artist='Drake', verbose=False)
syllables = result['syllables']
bpm = result['beat']['bpm']

# Load isolated vocals
y, sr = librosa.load('data/drake_vocals_isolated.wav', sr=22050, mono=True)
y_stereo, _ = librosa.load('data/drake_vocals_isolated.wav', sr=22050, mono=False)

baseline = get_main_vocal_baseline(y, sr, syllables)

# Detection
hop_length = 512
frame_length = 2048

rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

n_frames = len(rms)
width = np.zeros(n_frames)
left, right = y_stereo[0], y_stereo[1]

for i in range(min(n_frames, (len(left)-frame_length)//hop_length)):
    start = i * hop_length
    end = start + frame_length
    l_frame, r_frame = left[start:end], right[start:end]
    if np.std(l_frame) > 0 and np.std(r_frame) > 0:
        corr = np.corrcoef(l_frame, r_frame)[0, 1]
        width[i] = 1.0 - max(0, corr)

loud_mask = rms > np.percentile(rms, 10)
baseline_rms = np.mean(rms[loud_mask])
baseline_width = np.mean(width[loud_mask])

rms_db = 20 * np.log10(rms / (baseline_rms + 1e-10))
width_deviation = width - baseline_width

is_candidate = (rms_db > 5.0) | (width_deviation > 0.3)
is_candidate_smooth = uniform_filter1d(is_candidate.astype(float), size=5) > 0.5

candidates = []
in_region = False
start_idx = 0

for i, is_c in enumerate(is_candidate_smooth):
    if is_c and not in_region:
        in_region = True
        start_idx = i
    elif not is_c and in_region:
        in_region = False
        if times[i] - times[start_idx] >= 0.12:
            candidates.append((times[start_idx], times[i], start_idx, i))

# Find true doubles (lyric-aligned)
true_doubles = []
beat_vocals = []

for start_time, end_time, start_idx, end_idx in candidates:
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = y[start_sample:end_sample]

    if len(segment) < 512:
        continue

    _, pitch_match, spectral_sim, lyric_aligned = is_likely_beat_vocal(
        segment, sr, baseline, syllables, start_time, end_time
    )

    region_amp_db = np.mean(rms_db[start_idx:end_idx])
    region_width = np.mean(width[start_idx:end_idx])

    # Find the words
    words = []
    for syl in syllables:
        syl_start = syl.get('start', 0)
        syl_end = syl.get('end', 0)
        if syl_start < end_time and syl_end > start_time:
            words.append(syl.get('text', ''))
    word_str = ' '.join(words).strip()

    entry = {
        'start': start_time,
        'end': end_time,
        'amp_db': region_amp_db,
        'width': region_width,
        'words': word_str,
        'lyric_aligned': lyric_aligned,
        'pitch_match': pitch_match
    }

    if lyric_aligned and pitch_match > 0.5:
        true_doubles.append(entry)
    elif region_width > 0.5:  # High stereo width beat vocals
        beat_vocals.append(entry)

# Sort by amplitude (loudest first)
true_doubles.sort(key=lambda x: -x['amp_db'])
beat_vocals.sort(key=lambda x: -x['width'])

# Create output directories
output_dir = Path('data/true_doubles')
output_dir.mkdir(exist_ok=True)

beat_vocal_dir = Path('data/beat_vocal_examples')
beat_vocal_dir.mkdir(exist_ok=True)

print('EXPORTING TRUE DOUBLES (lyric-aligned)')
print('=' * 60)

padding = 0.5  # seconds before/after

for i, d in enumerate(true_doubles, 1):
    start_sec = max(0, d['start'] - padding)
    end_sec = min(len(y_stereo[0])/sr, d['end'] + padding)

    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)

    clip = y_stereo[:, start_sample:end_sample]

    # Create filename
    words_safe = d['words'][:20].replace(' ', '_').replace('/', '-').replace("'", '').replace('"', '')
    time_str = f"{int(d['start']//60)}m{int(d['start']%60):02d}s"
    filename = f"double_{i:02d}_{time_str}_{words_safe}.wav"
    filepath = output_dir / filename

    sf.write(str(filepath), clip.T, sr)

    print(f"  {i:2d}. {d['start']:6.2f}s | +{d['amp_db']:5.1f}dB | w={d['width']:.2f} | \"{d['words'][:25]}\"")

print()
print(f'Exported {len(true_doubles)} true doubles to: data/true_doubles/')
print()

print('EXPORTING BEAT VOCAL EXAMPLES (for comparison)')
print('=' * 60)

for i, d in enumerate(beat_vocals[:5], 1):
    start_sec = max(0, d['start'] - padding)
    end_sec = min(len(y_stereo[0])/sr, d['end'] + padding)

    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)

    clip = y_stereo[:, start_sample:end_sample]

    time_str = f"{int(d['start']//60)}m{int(d['start']%60):02d}s"
    filename = f"beat_vocal_{i:02d}_{time_str}_width{d['width']:.2f}.wav"
    filepath = beat_vocal_dir / filename

    sf.write(str(filepath), clip.T, sr)

    print(f"  {i:2d}. {d['start']:6.2f}s | width={d['width']:.2f} | NOT lyric-aligned")

print()
print(f'Exported {min(5, len(beat_vocals))} beat vocal examples to: data/beat_vocal_examples/')
print()
print('TO VERIFY:')
print('  - True doubles should sound like Drake emphasizing words')
print('  - Beat vocals should sound like samples/effects (stereo, not Drake)')
