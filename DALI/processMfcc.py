import os
import sys
import signal
import librosa
import soundfile as sf
import torch
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import random

import DALI as dali
from demucs.pretrained import get_model
from demucs.apply import apply_model

# CONFIG


mp.set_start_method("spawn", force=True)

STOP = mp.Value('b', False)

BASE = os.path.join(os.path.dirname(__file__), "DALI_v1.0")
ANNOTATION_PATH = os.path.join(BASE, "annotations")
AUDIO_FOLDER = os.path.join(BASE, "audio")

OUTPUT_TRAINING_FOLDER = os.path.join(BASE, "processed-training")
OUTPUT_TESTING_FOLDER = os.path.join(BASE, "processed-testing")

TRAIN_METADATA_OUT = os.path.join(OUTPUT_TRAINING_FOLDER, "metadata")
TRAIN_POLY_OUT = os.path.join(OUTPUT_TRAINING_FOLDER, "polyphonic")
TRAIN_VOCAL_OUT = os.path.join(OUTPUT_TRAINING_FOLDER, "vocals")
TRAIN_SEGMENT_OUT = os.path.join(OUTPUT_TRAINING_FOLDER, "segments")

TEST_METADATA_OUT = os.path.join(OUTPUT_TESTING_FOLDER, "metadata")
TEST_POLY_OUT = os.path.join(OUTPUT_TESTING_FOLDER, "polyphonic")
TEST_VOCAL_OUT = os.path.join(OUTPUT_TESTING_FOLDER, "vocals")
TEST_SEGMENT_OUT = os.path.join(OUTPUT_TESTING_FOLDER, "segments")

global DEVICE, DEMUCS_MODEL

for d in [
    OUTPUT_TRAINING_FOLDER, TRAIN_POLY_OUT, TRAIN_VOCAL_OUT, TRAIN_SEGMENT_OUT, TRAIN_METADATA_OUT,
    OUTPUT_TESTING_FOLDER, TEST_POLY_OUT, TEST_VOCAL_OUT, TEST_SEGMENT_OUT, TEST_METADATA_OUT,
]:
    os.makedirs(d, exist_ok=True)

TARGET_SR = 16000
BATCH_SIZE = 4
NUM_WORKERS = 8
QUEUE_MAXSIZE = 16

TRAIN_SPLIT = 0.95
SPLIT_SEED = 42



SENTINEL = None

# SIGNAL HANDLING

def handle_sigint(signum, frame):
    print("\n[STOP REQUESTED]")
    STOP.value = True

signal.signal(signal.SIGINT, handle_sigint)

# SPLITS

def assign_splits(items, train_ratio=TRAIN_SPLIT, seed=SPLIT_SEED):
    rng = random.Random(seed)
    ids = [song_id for song_id, _ in items]
    rng.shuffle(ids)
    split_idx = int(len(ids) * train_ratio)
    return set(ids[:split_idx]), set(ids[split_idx:])


def get_dirs_for_split(song_id, train_set):
    if song_id in train_set:
        return TRAIN_POLY_OUT, TRAIN_VOCAL_OUT, TRAIN_SEGMENT_OUT, TRAIN_METADATA_OUT
    else:
        return TEST_POLY_OUT, TEST_VOCAL_OUT, TEST_SEGMENT_OUT, TEST_METADATA_OUT

# CACHE CHECKS


def metadata_exists(song_id, metadata_out):
    return os.path.exists(os.path.join(metadata_out, f"{song_id}.pt"))

# DEMUCS


def run_demucs_batch(batch_items):
    max_len = max(len(audio) for _, audio in batch_items)

    batch = []
    for _, audio in batch_items:
        audio = torch.tensor(audio).float()
        audio = audio.unsqueeze(0).repeat(2, 1)
        audio = torch.nn.functional.pad(audio, (0, max_len - audio.shape[1]))
        batch.append(audio)

    batch = torch.stack(batch).to(DEVICE)

    with torch.no_grad():
        sources = apply_model(DEMUCS_MODEL, batch, device=DEVICE)

    return sources

# PRODUCER (GPU)


def gpu_producer(dataset, train_set, queue, limit=None):
    items = list(dataset.items())
    if limit:
        items = items[:limit]

    batch = []

    for song_id, entry in tqdm(items, desc="GPU Producer"):
        if STOP.value:
            break

        poly_out, vocal_out, _, metadata_out = get_dirs_for_split(song_id, train_set)

        # FULL CACHE SKIP
        if metadata_exists(song_id, metadata_out):
            print(f"[FULL CACHE] {song_id}")
            continue

        youtube_id = entry.info["audio"]["url"]
        audio_path = os.path.join(AUDIO_FOLDER, youtube_id + ".wav")

        if not os.path.exists(audio_path):
            print(f"[MISSING AUDIO] {audio_path}")
            continue

        audio, _ = librosa.load(audio_path, sr=TARGET_SR)
        batch.append((song_id, entry, audio))

        if len(batch) == BATCH_SIZE:
            sources = run_demucs_batch([(sid, a) for sid, _, a in batch])

            for i, (sid, entry, audio) in enumerate(batch):
                vocals = sources[i, 3].cpu().numpy().T

                sf.write(os.path.join(poly_out, f"{sid}.wav"), audio, TARGET_SR)
                sf.write(os.path.join(vocal_out, f"{sid}.wav"), vocals, TARGET_SR)

                queue.put((sid, entry))

            batch = []

    # flush remaining
    if batch:
        sources = run_demucs_batch([(sid, a) for sid, _, a in batch])
        for i, (sid, entry, audio) in enumerate(batch):
            vocals = sources[i, 3].cpu().numpy().T

            poly_out, vocal_out, _, _ = get_dirs_for_split(sid, train_set)

            sf.write(os.path.join(poly_out, f"{sid}.wav"), audio, TARGET_SR)
            sf.write(os.path.join(vocal_out, f"{sid}.wav"), vocals, TARGET_SR)

            queue.put((sid, entry))

    # send stop signals
    for _ in range(NUM_WORKERS):
        queue.put(SENTINEL)

# MFCC

def extract_mfcc(signal):
    if len(signal) < 400:
        signal = librosa.util.fix_length(signal, size = 400)

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=TARGET_SR,
        n_mfcc=40,
        n_fft=400,
        hop_length=160,
        win_length=400
    )

    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-5)
    return torch.tensor(mfcc)

# CONSUMER (CPU)

def process_song_cpu(song_id, entry, train_set):
    poly_out, vocal_out, segment_out, metadata_out = get_dirs_for_split(song_id, train_set)

    if metadata_exists(song_id, metadata_out):
        print(f"[CACHE CPU] {song_id}")
        return

    audio_path = os.path.join(poly_out, f"{song_id}.wav")
    vocal_path = os.path.join(vocal_out, f"{song_id}.wav")

    if not os.path.exists(audio_path) or not os.path.exists(vocal_path):
        print(f"[MISSING STEMS] {song_id}")
        return

    audio, _ = librosa.load(audio_path, sr=TARGET_SR)
    vocals, _ = librosa.load(vocal_path, sr=TARGET_SR)

    words = entry.annotations["annot"]["words"]
    segments = []

    for i, w in enumerate(words):
        start, end = w["time"]

        if start is None or end is None or end <= start:
            continue

        s, e = int(start * TARGET_SR), int(end * TARGET_SR)
        seg_audio = audio[s:e]
        seg_vocals = vocals[s:e]

        if len(seg_audio) == 0:
            continue

        seg_dir = os.path.join(segment_out, song_id)
        os.makedirs(seg_dir, exist_ok=True)

        sf.write(os.path.join(seg_dir, f"{i:04d}_poly.wav"), seg_audio, TARGET_SR)
        sf.write(os.path.join(seg_dir, f"{i:04d}_voc.wav"), seg_vocals, TARGET_SR)

        mfcc_poly = extract_mfcc(seg_audio)
        mfcc_voc = extract_mfcc(seg_vocals)

        T = min(mfcc_poly.shape[1], mfcc_voc.shape[1])
        mfcc_poly = mfcc_poly[:, :T]
        mfcc_voc = mfcc_voc[:, :T]

        segments.append({
            "text": w["text"],
            "start": start,
            "end": end,
            "mfcc_poly": mfcc_poly,
            "mfcc_voc": mfcc_voc,
        })

    if segments:
        torch.save({
            "song_id": song_id,
            "segments": segments,
        }, os.path.join(metadata_out, f"{song_id}.pt"))

        print(f"[DONE] {song_id}")

def cpu_consumer(queue, train_set):
    while True:
        try:
            if STOP.value:
                break

            item = queue.get()

            if item is SENTINEL:
                break

            song_id, entry = item
            process_song_cpu(song_id, entry, train_set)
        except Exception as e:
            print(f"[ERROR] {song_id}: {e}")

# MAIN

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None


    DEVICE = "mps" if torch.mps.is_available() else "cpu"
    DEVICE = "cuda" if torch.cuda.is_available() else DEVICE
    print(f"Using device: {DEVICE}")

    print("[INIT] Loading Demucs model...")
    DEMUCS_MODEL = get_model("htdemucs").to(DEVICE)
    DEMUCS_MODEL.eval()

    dataset = dali.get_the_DALI_dataset(ANNOTATION_PATH)
    items = list(dataset.items())

    train_set, _ = assign_splits(items)

    queue = mp.Queue(maxsize=QUEUE_MAXSIZE)

    consumers = []
    for _ in range(NUM_WORKERS):
        p = mp.Process(target=cpu_consumer, args=(queue, train_set))
        p.start()
        consumers.append(p)

    gpu_producer(dataset, train_set, queue, limit)

    for p in consumers:
        p.join()

    print("Pipeline complete.")