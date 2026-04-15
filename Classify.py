import os
import torch
import pandas as pd
from scipy import stats
from collections import defaultdict
from transformers import pipeline
from tqdm import tqdm

# Ensure these helper functions/classes are accessible
from evaluate import ctc_beam_search, decode_output, BLANK
from train import DualCNNBiLSTM, LyricsDataset, decode_output, NUM_CLASSES, BLANK
from model import score_with_lm, load_lm

# --- 1. Configuration & Paths ---
PROCESSED_FOLDER = "DALI/DALI_v1.0/processed-testing/metadata"
VOCALS_DIR = "DALI/DALI_v1.0/processed-testing/vocals"
MODEL_PATH = "test/dual_cnn_bilstm_ctcn10.pt" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Helper Functions ---
def word_error_rate(reference, hypothesis):
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    r, h = len(ref_words), len(hyp_words)
    if r == 0: return h 
    
    dp = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1): dp[i][0] = i
    for j in range(h + 1): dp[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[r][h] / r

# --- 3. Initialize Classifiers & Models ---
print("Initializing Genre Classifier...")
genre_classifier = pipeline(
    "audio-classification", 
    model="m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres", 
    device=0 if torch.cuda.is_available() else -1
)

unigram_lm, bigram_lm = load_lm("lm.pt")

print(f"Loading model: {MODEL_PATH}")
model = DualCNNBiLSTM(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

dataset = LyricsDataset(PROCESSED_FOLDER)

# --- 4. Main Processing Loop ---
genre_buckets = defaultdict(list)
results_log = []

meta_files = sorted([f for f in os.listdir(PROCESSED_FOLDER) if f.endswith('.pt')])

print(f"Starting Evaluation on {len(meta_files)} files...")

with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        
        mfcc_poly, mfcc_voc, lyrics = dataset[i]
        if lyrics is None or len(lyrics.strip()) == 0:
            continue
        
        song_id = meta_files[0].replace(".pt", "") 
        
        # Get Genre
        audio_path = os.path.join(VOCALS_DIR, f"{song_id}.wav")
        genre = "Unknown"
        if os.path.exists(audio_path):
            try:
                res = genre_classifier(audio_path, top_k=1)
                genre = res[0]['label']
            except:
                pass
        
        # Run Transcription Inference
        mfcc_poly = mfcc_poly.unsqueeze(0).to(device)
        mfcc_voc = mfcc_voc.unsqueeze(0).to(device)
        
        # Model output shape is [T, B, C] based on your train.py permute
        out = model(mfcc_poly, mfcc_voc)  # [1, time_steps, num_classes]
        pred_indices = out.argmax(dim=2)[:, 0].cpu().tolist()
        predicted = decode_output(pred_indices)
        
        wer = word_error_rate(lyrics, predicted)
        
        if genre != "Unknown":
            genre_buckets[genre].append(wer)
            results_log.append({"Song_ID": song_id, "Genre": genre, "WER": wer})

# --- 5. Kruskal-Wallis Test ---
groups = [scores for g, scores in genre_buckets.items() if len(scores) >= 5]
if len(groups) > 1:
    h_stat, p_val = stats.kruskal(*groups)
    print(f"\nKruskal-Wallis H-test: H={h_stat:.4f}, p={p_val:.4e}")
    
    stats_df = pd.DataFrame([
        {"Genre": g, "Avg WER": sum(s)/len(s), "Count": len(s)} 
        for g, s in genre_buckets.items() if len(s) >= 5
    ]).sort_values("Avg WER")
    print("\n", stats_df.to_string(index=False))

pd.DataFrame(results_log).to_csv("genre_impact_final.csv", index=False)