import os
import torch
from collections import defaultdict
from train import LyricsDataset

# Build Language Models


def build_unigram_lm(dataset):
    word_counts = defaultdict(int)
    total_words = 0

    for _, _, lyrics in dataset:
        for word in lyrics.split():
            word_counts[word] += 1
            total_words += 1

    lm = {}
    for word, count in word_counts.items():
        lm[word] = torch.log(torch.tensor(count / total_words)).item()

    return lm


def build_bigram_lm(dataset):
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)

    for _, _, lyrics in dataset:
        words = lyrics.split()

        for i in range(len(words)):
            unigram_counts[words[i]] += 1
            if i > 0:
                bigram = (words[i - 1], words[i])
                bigram_counts[bigram] += 1

    lm = {}
    for (w1, w2), count in bigram_counts.items():
        lm[(w1, w2)] = torch.log(
            torch.tensor(count / unigram_counts[w1])
        ).item()

    return lm

# LM Scoring


def score_with_lm(text, unigram_lm, bigram_lm=None):
    words = text.split()
    score = 0.0

    for i, w in enumerate(words):
        # unigram
        score += unigram_lm.get(w, -10.0)

        # bigram
        if bigram_lm and i > 0:
            score += bigram_lm.get((words[i - 1], w), -5.0)

    return score


# Save / Load


def save_lm(unigram_lm, bigram_lm, path="lm.pt"):
    torch.save({
        "unigram": unigram_lm,
        "bigram": bigram_lm
    }, path)


def load_lm(path="lm.pt"):
    data = torch.load(path)
    return data["unigram"], data["bigram"]


if __name__ == "__main__":
    PROCESSED_FOLDER = "DALI/DALI_v1.0/processed-training/metadata"
    dataset = LyricsDataset(PROCESSED_FOLDER)
    unigram_lm = build_unigram_lm(dataset)
    bigram_lm = build_bigram_lm(dataset)
    save_lm(unigram_lm, bigram_lm)
