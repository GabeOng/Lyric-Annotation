import os
import torch
import librosa
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor
import DALI as dali
from demucs import pretrained
from demucs.apply import apply_model
import signal
import multiprocessing as mp
import numpy as np
import sys


## TODO: PARALLELIZE CPU TO RUN AS SOON AS DEMUCS PRODUCES FIRST BATCH OF STEMS (CURRENTLY CPU STARTS AFTER ALL DEMUCS DONE)
# -------------------------
# CONFIG
# -------------------------
STOP = mp.Value('b', False)  # Shared flag for graceful shutdown

BASE = os.path.join(os.path.dirname(__file__), "DALI_v1.0/")
ANNOTATION_PATH = os.path.join(BASE, "annotations")
AUDIO_FOLDER = os.path.join(BASE, "audio")
OUTPUT_FOLDER = os.path.join(BASE, "processed")

METADATA_OUT = os.path.join(OUTPUT_FOLDER, "metadata")
POLY_OUT = os.path.join(OUTPUT_FOLDER, "polyphonic")
VOCAL_OUT = os.path.join(OUTPUT_FOLDER, "vocals")
SEGMENT_OUT = os.path.join(OUTPUT_FOLDER, "segments")



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

DEMUCS_MODEL = pretrained.get_model('htdemucs')
DEMUCS_MODEL.to(DEVICE)
DEMUCS_MODEL.eval()

for d in [OUTPUT_FOLDER, POLY_OUT, VOCAL_OUT, SEGMENT_OUT, METADATA_OUT]:
    os.makedirs(d, exist_ok=True)

TARGET_SR = 16000


# -------------------------
# DEMUCS SEPARATION
# -------------------------

# if stems already exist, skip GPU separation step
def stems_exist(song_id):
    return(
        os.path.exists(os.path.join(POLY_OUT, f"{song_id}.wav")) and
        os.path.exists(os.path.join(VOCAL_OUT, f"{song_id}.wav"))
    )


# Extract vocals using Demucs. Returns (samples, 2) stereo array. Handles mono→stereo conversion and GPU processing.
def extract_vocals(audio, sr):
    # mono → stereo
    x = torch.tensor(audio).float().unsqueeze(0)  # (1, samples)
    x = x.repeat(2, 1).unsqueeze(0)               # (1, 2, samples)
    x = x.to(DEVICE)

    with torch.no_grad():
        sources = apply_model(DEMUCS_MODEL, x, device=DEVICE)[0]

    vocals = sources[3].cpu().numpy().T   # (samples, 2)
    return vocals

#Batch processing for demucs
def run_demucs_batch(batch_items):
    """
    batch_items = list of (song_id, audio_array)
    """
    # Convert mono → stereo and pad to same length
    max_len = max(len(audio) for _, audio in batch_items)

    batch = []
    for _, audio in batch_items:
        audio = torch.tensor(audio).float()
        audio = audio.unsqueeze(0).repeat(2, 1)  # (2, samples)
        audio = torch.nn.functional.pad(audio, (0, max_len - audio.shape[1]))
        batch.append(audio)

    batch = torch.stack(batch, dim=0).to(DEVICE)  # (B, 2, max_len)

    with torch.no_grad():
        sources = apply_model(DEMUCS_MODEL, batch, device=DEVICE)

    # sources shape: (B, 4, 2, samples)
    return sources

def process_demucs_batch(batch_items):
    sources = run_demucs_batch(batch_items)

    for i, (song_id, audio) in enumerate(batch_items):
        vocals = sources[i, 3].cpu().numpy().T  # (samples, 2)

        # Save polyphonic (original)
        sf.write(os.path.join(POLY_OUT, f"{song_id}.wav"), audio, TARGET_SR)

        # Save vocals
        sf.write(os.path.join(VOCAL_OUT, f"{song_id}.wav"), vocals, TARGET_SR)

        print(f"[STEMS] {song_id}")

BATCH_SIZE = 4

def run_demucs_pass(dataset):
    items = list(dataset.items())
    batch = []

    for song_id, entry in items:
        if STOP.value:
            print("[STOP] Halting Demucs pass.")
            return

        if stems_exist(song_id):
            print(f"[CACHE] {song_id}")
            continue

        youtube_id = entry.info["audio"]["url"]
        audio_path = os.path.join(AUDIO_FOLDER, youtube_id + ".wav")

        if not os.path.exists(audio_path):
            print(f"[MISSING AUDIO] {audio_path}")
            continue

        audio, _ = librosa.load(audio_path, sr=TARGET_SR)
        batch.append((song_id, audio))

        if len(batch) == BATCH_SIZE:
            process_demucs_batch(batch)
            batch = []

    if batch and not STOP.value:
        process_demucs_batch(batch)


# -------------------------
# MFCC EXTRACTION
# -------------------------
def extract_mfcc(signal, sr=16000, n_mfcc=40):
    # Pad short signals so n_fft fits
    if len(signal) < 400:
        signal = librosa.util.fix_length(signal, 400)

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=400,        # 25 ms window
        hop_length=160,   # 10 ms hop
        win_length=400
    )
    # Normalize MFCCs (per song)
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-5)
    return torch.tensor(mfcc)


# -------------------------
# PROCESS SONG
# -------------------------
def process_song_gpu(args):
    song_id, entry = args

    try:
        youtube_id = entry.info["audio"]["url"]
        audio_path = os.path.join(AUDIO_FOLDER, youtube_id + ".wav")

        if not os.path.exists(audio_path):
            print(f"[MISSING] {audio_path}")
            return None

        # -------------------------
        # LOAD AUDIO
        # -------------------------
        audio, sr = librosa.load(audio_path, sr=TARGET_SR)

        # -------------------------
        # VOCAL EXTRACTION
        # -------------------------
        vocals = extract_vocals(audio, TARGET_SR)

        # Save raw polyphonic + vocal audio
        sf.write(os.path.join(POLY_OUT, f"{song_id}.wav"), audio, TARGET_SR)
        sf.write(os.path.join(VOCAL_OUT, f"{song_id}.wav"), vocals, TARGET_SR)

        # -------------------------
        # SEGMENT BY LYRICS
        # -------------------------
        words = entry.annotations["annot"]["words"]

        segments = []
        for i, w in enumerate(words):
            start = w["time"][0]
            if start is None or end is None:
                continue
            if start < 0 or end <= start:
                continue
            end = w["time"][1]
            text = w["text"]

            s = int(start * TARGET_SR)
            e = int(end * TARGET_SR)

            if e <= s:
                continue

            seg_audio = audio[s:e]
            seg_vocals = vocals[s:e]

            seg_dir = os.path.join(SEGMENT_OUT, song_id)
            os.makedirs(seg_dir, exist_ok=True)

            seg_path_poly = os.path.join(seg_dir, f"{i:04d}_poly.wav")
            seg_path_voc = os.path.join(seg_dir, f"{i:04d}_voc.wav")

            sf.write(seg_path_poly, seg_audio, TARGET_SR)
            sf.write(seg_path_voc, seg_vocals, TARGET_SR)

            # -------------------------
            # MFCC EXTRACTION
            # -------------------------
            if len(seg_audio) == 0 or len(seg_vocals) == 0:
                print(f"[EMPTY STEM] {song_id}")
                return
            if not np.isfinite(mfcc_poly).all() or not np.isfinite(mfcc_voc).all():
                print(f"[BAD AUDIO] {song_id}")
                return
            mfcc_poly = extract_mfcc(seg_audio, sr=TARGET_SR)
            mfcc_voc = extract_mfcc(seg_vocals, sr=TARGET_SR)

            segments.append({
                "index": i,
                "text": text,
                "start": start,
                "end": end,
                "mfcc_poly": mfcc_poly,
                "mfcc_voc": mfcc_voc,
            })

        # -------------------------
        # SAVE METADATA
        # -------------------------
        torch.save(
            {
                "song_id": song_id,
                "youtube_id": youtube_id,
                "segments": segments,
            },
            os.path.join(METADATA_OUT, f"{song_id}.pt")
        )

        print(f"[OK] {song_id}")
        return song_id

    except Exception as e:
        print(f"[ERROR] {song_id}: {e}")
        return None
    
def process_song_cpu(args):
    if STOP.value: 
        return None
    song_id, entry = args

    try:
        youtube_id = entry.info["audio"]["url"]
        audio_path = os.path.join(POLY_OUT, f"{song_id}.wav")
        vocal_path = os.path.join(VOCAL_OUT, f"{song_id}.wav")

        if not (os.path.exists(audio_path) and os.path.exists(vocal_path)):
            print(f"[MISSING STEMS] {song_id}")
            return None

        audio, _ = librosa.load(audio_path, sr=TARGET_SR)
        vocals, _ = librosa.load(vocal_path, sr=TARGET_SR)

        words = entry.annotations["annot"]["words"]
        segments = []

        for i, w in enumerate(words):
            start = w["time"][0]
            end = w["time"][1]
            text = w["text"]

            if start is None or end is None:
                print(f"[BAD TIME] {song_id} word {i}: {w}")
                continue
            if start < 0 or end <= start:
                print(f"[BAD RANGE] {song_id} word {i}: {w}")
                continue

            s = int(start * TARGET_SR)
            e = int(end * TARGET_SR)
            if e <= s:
                continue

            seg_audio = audio[s:e]
            seg_vocals = vocals[s:e]

            seg_dir = os.path.join(SEGMENT_OUT, song_id)
            os.makedirs(seg_dir, exist_ok=True)

            seg_path_poly = os.path.join(seg_dir, f"{i:04d}_poly.wav")
            seg_path_voc = os.path.join(seg_dir, f"{i:04d}_voc.wav")

            sf.write(seg_path_poly, seg_audio, TARGET_SR)
            sf.write(seg_path_voc, seg_vocals, TARGET_SR)

            mfcc_poly = extract_mfcc(seg_audio, sr=TARGET_SR)
            mfcc_voc = extract_mfcc(seg_vocals, sr=TARGET_SR)

            segments.append({
                "index": i,
                "text": text,
                "start": start,
                "end": end,
                "mfcc_poly": mfcc_poly,
                "mfcc_voc": mfcc_voc,
            })

        torch.save(
            {
                "song_id": song_id,
                "youtube_id": youtube_id,
                "segments": segments,
            },
            os.path.join(METADATA_OUT, f"{song_id}.pt")
        )

        print(f"[OK CPU] {song_id}")
        return song_id

    except Exception as e:
        print(f"[ERROR CPU] {song_id}: {e}")
        return None


def handle_sigint(signum, frame):
    print("\n[STOP REQUESTED] Finishing current tasks...")
    STOP.value = True

signal.signal(signal.SIGINT, handle_sigint)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    limit = None
    if len(sys.argsv[1]) > 1:
        try:
            limit = int(sys.argv[1])
            if limit > 0:
                print(f"Processing only first {limit} songs (for testing)")
            else:
                limit = None
        except ValueError:
            print("Invalid argument. Expected an integer.")


    dataset = dali.get_the_DALI_dataset(ANNOTATION_PATH)
    items = list(dataset.items())

    print(f"Total songs: {len(items)}")

    # --------- PASS 1: Demucs (GPU, single process) ---------
    print("Running Demucs separation pass...")
    run_demucs_pass(dataset)

    # --------- PASS 2: Segmentation + MFCC (CPU, parallel) ---------
    tasks = [(song_id, entry) for song_id, entry in items]
    
    print(f"Processing {len(tasks)} songs in CPU pool...")

    
    try:
        with ProcessPoolExecutor(max_workers=8) as ex: # <- Adjust number of workers based on your CPU cores
            for _ in ex.map(process_song_cpu, tasks[:limit] if limit else tasks):
                pass
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Stopping processing...")
