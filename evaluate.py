import torch
import os
from train import CNN, LyricsDataset, decode_output, NUM_CLASSES

PROCESSED_FOLDER = "DALI/DALI_v1.0/processed"
MODEL_PATH = "model.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# WER calc
def word_error_rate(reference, hypothesis):
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    r, h = len(ref_words), len(hyp_words)

    # dynamic programming matrix
    dp = [[0] * (h + 1) for _ in range(r + 1)]

    for i in range(r + 1):
        dp[i][0] = i
    for j in range(h + 1):
        dp[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )

    wer = dp[r][h] / max(r, 1)
    return wer

#Load model
model = CNN(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Load dataset
dataset = LyricsDataset(PROCESSED_FOLDER)

total_wer = 0.0

with torch.no_grad():
    for i in range(len(dataset)):
        mfcc_poly, mfcc_voc, lyrics = dataset[i]
        if lyrics is None:
            continue
        mfcc_poly = mfcc_poly.unsqueeze(0).to(device)  # [1, n_mfcc, time_steps]
        mfcc_voc = mfcc_voc.unsqueeze(0).to(device)    # [1, n_mfcc, time_steps]

        out = model(mfcc_poly, mfcc_voc)  # [1, time_steps, num_classes]
        pred_indices = out.argmax(dim=2).squeeze(0).cpu().tolist()
        predicted = decode_output(pred_indices)
        
        wer = word_error_rate(lyrics, predicted)
        total_wer += wer

        print(f"--- Song {i+1} ---")
        print(f"Target:    {lyrics[:80]}")
        print(f"Predicted: {predicted[:80]}")
        print(f"WER:       {wer:.2%}")
        print()

avg_wer = total_wer / len(dataset)
print(f"Average WER across {len(dataset)} songs: {avg_wer:.2%}")