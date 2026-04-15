import torch
import os
from train import DualCNNBiLSTM, LyricsDataset, decode_output, NUM_CLASSES, BLANK
from model import score_with_lm, load_lm
from collections import defaultdict

PROCESSED_FOLDER = "DALI/DALI_v1.0/processed-testing/metadata"
MODEL_FOLDER = "dprctd_pt/"
BASE_FOLDER = os.getcwd()
print(BASE_FOLDER)
folders = sorted([
    os.path.join(MODEL_FOLDER, f) 
    for f in os.listdir(MODEL_FOLDER)

])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ctc_beam_search(log_probs, unigram_lm, bigram_lm=None, beam_width=10, lm_weight=0.5):
    beams = {(): (0.0, float('-inf'))}
    T, C = log_probs.shape

    for t in range(T):
        new_beams = defaultdict(lambda: (float('-inf'), float('-inf')))

        for prefix, (p_b, p_nb) in beams.items():
            for c in range(C):
                p = log_probs[t, c].item()

                if c == BLANK:
                    nb_pb, nb_pnb = new_beams[prefix]
                    nb_pb = torch.logaddexp(torch.tensor(nb_pb), torch.tensor(p_b + p))
                    nb_pb = torch.logaddexp(nb_pb, torch.tensor(p_nb + p))
                    new_beams[prefix] = (nb_pb.item(), nb_pnb)

                else:
                    new_prefix = prefix + (c,)
                    nb_pb, nb_pnb = new_beams[new_prefix]

                    if prefix and prefix[-1] == c:
                        nb_pnb = torch.logaddexp(
                            torch.tensor(nb_pnb),
                            torch.tensor(p_b + p)
                        ).item()
                    else:
                        nb_pnb = torch.logaddexp(
                            torch.tensor(nb_pnb),
                            torch.tensor(p_b + p)
                        )
                        nb_pnb = torch.logaddexp(
                            nb_pnb,
                            torch.tensor(p_nb + p)
                        ).item()

                    new_beams[new_prefix] = (nb_pb, nb_pnb)

        # prune
        beams = sorted(
            new_beams.items(),
            key=lambda x: torch.logaddexp(
                torch.tensor(x[1][0]),
                torch.tensor(x[1][1])
            ).item(),
            reverse=True
        )[:beam_width]

        beams = dict(beams)

    # apply LM scoring here
    best_score = float('-inf')
    best_text = ""

    for prefix, (p_b, p_nb) in beams.items():
        acoustic_score = torch.logaddexp(
            torch.tensor(p_b), torch.tensor(p_nb)
        ).item()

        text = decode_output(prefix)

        lm_score = score_with_lm(text, unigram_lm, bigram_lm)

        total_score = acoustic_score + lm_weight * lm_score

        if total_score > best_score:
            best_score = total_score
            best_text = text

    return best_text


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



# Load dataset
dataset = LyricsDataset(PROCESSED_FOLDER)

uunigram_lm, bigram_lm = load_lm("lm.pt")

print(folders)

for models in folders:
    total_edits = 0
    total_words = 0
    count = 0

    # Load model
    model = DualCNNBiLSTM(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(models, map_location=device))
    model.eval()

    with torch.no_grad():
        for i in range(len(dataset)):
            mfcc_poly, mfcc_voc, lyrics = dataset[i]
            if lyrics is None:
                continue
            mfcc_poly = mfcc_poly.unsqueeze(0).to(device)
            mfcc_voc = mfcc_voc.unsqueeze(0).to(device)

            out = model(mfcc_poly, mfcc_voc)
            log_probs = out[:, 0, :]
            predicted = ctc_beam_search(log_probs, beam_width=10, unigram_lm=uunigram_lm, bigram_lm=bigram_lm, lm_weight=0.5)

            # Corpus-level edit accumulation
            ref_words = lyrics.lower().split()
            hyp_words = predicted.lower().split()
            r, h = len(ref_words), len(hyp_words)

            dp = [[0] * (h + 1) for _ in range(r + 1)]
            for ii in range(r + 1): dp[ii][0] = ii
            for jj in range(h + 1): dp[0][jj] = jj
            for ii in range(1, r + 1):
                for jj in range(1, h + 1):
                    if ref_words[ii - 1] == hyp_words[jj - 1]:
                        dp[ii][jj] = dp[ii - 1][jj - 1]
                    else:
                        dp[ii][jj] = 1 + min(dp[ii - 1][jj], dp[ii][jj - 1], dp[ii - 1][jj - 1])

            total_edits += dp[r][h]
            total_words += max(r, 1)
            count += 1

            print(f"--- Segment {i+1} ---")
            print(f"Target:    {lyrics[:80]}")
            print(f"Predicted: {predicted[:80]}")
            print()

    corpus_wer = total_edits / total_words
    print(f"Model: {models}")
    print(f"Corpus WER across {count} segments: {corpus_wer:.2%}")
    print()

    # Write Results
    out_path = os.path.join("results", f"greedyresults_{os.path.basename(models)}.txt")
    with open(out_path, "w") as f:
        f.write(f"Average WER across {len(dataset)} segments: {corpus_wer:.2%}\n")