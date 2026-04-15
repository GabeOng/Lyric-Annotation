import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import re 

# Vocabulary

EPOCHS = 20
CHARS = "abcdefghijklmnopqrstuvwxyz' "
BLANK = 0
char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = CTC blank
idx_to_char = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1

def encode_lyrics(lyrics: str):
    lyrics = lyrics.lower()
    return [char_to_idx[c] for c in lyrics if c in char_to_idx]

def normalize_text(text: str):
    text = text.lower()

    # remove punctuation (keep letters, space, apostrophe if you want)
    text = re.sub(r"[^a-z' ]+", "", text)

    # collapse whitespace
    text = " ".join(text.strip().split())

    return text

def decode_output(indices):
    # greedy CTC decode: collapse repeats, drop blanks
    result = []
    prev = None

    for idx in indices:
        if idx != BLANK and idx != prev:
            char = idx_to_char.get(idx, "?")
            result.append(char)
        prev = idx

    text = "".join(result)

    # normalize whitespace
    text = normalize_text(text)

    return text

# ───────────────────────────────────────────────────────────────
# Dataset (segment-level, using metadata .pt)
# ───────────────────────────────────────────────────────────────

class LyricsDataset(Dataset):
    """
    Expects metadata .pt files like:
        {
            "song_id": ...,
            "segments": [
                {
                    "text": str,
                    "mfcc_poly": Tensor[n_mfcc, T],
                    "mfcc_voc":  Tensor[n_mfcc, T],
                    ...
                },
                ...
            ]
        }
    """

    def __init__(self, folder: str):
        self.segment_items = []

        for fname in os.listdir(folder):
            if not fname.endswith(".pt"):
                continue

            path = os.path.join(folder, fname)
            data = torch.load(path)

            segments = data.get("segments", [])
            if not segments:
                print(f"[WARN] No segments in {fname}, skipping.")
                continue

            for seg in segments:
                if ("mfcc_poly" not in seg or
                    "mfcc_voc" not in seg or
                    "text" not in seg):
                    print(f"[WARN] Bad segment in {fname}, skipping.")
                    continue

                self.segment_items.append({
                    "mfcc_poly": seg["mfcc_poly"].float(),
                    "mfcc_voc": seg["mfcc_voc"].float(),
                    "lyrics": seg["text"].lower()
                })

        print(f"[DATA] Loaded {len(self.segment_items)} valid segments.")

    def __len__(self):
        return len(self.segment_items)

    def __getitem__(self, idx):
        item = self.segment_items[idx]
        return item["mfcc_poly"], item["mfcc_voc"], item["lyrics"]


# ───────────────────────────────────────────────────────────────
# Collate (pads both streams, builds CTC lengths)
# ───────────────────────────────────────────────────────────────

def collate_fn(batch):
    mfcc_poly_list, mfcc_voc_list, lyrics_list = zip(*batch)

    # lengths per sample (time dimension)
    poly_lengths = torch.tensor([m.shape[1] for m in mfcc_poly_list], dtype=torch.long)
    voc_lengths = torch.tensor([m.shape[1] for m in mfcc_voc_list], dtype=torch.long)

    # pad poly: [B, n_mfcc, T_max]
    poly_padded = torch.nn.utils.rnn.pad_sequence(
        [m.T for m in mfcc_poly_list], batch_first=True
    ).permute(0, 2, 1)

    # pad voc: [B, n_mfcc, T_max]
    voc_padded = torch.nn.utils.rnn.pad_sequence(
        [m.T for m in mfcc_voc_list], batch_first=True
    ).permute(0, 2, 1)

    # encode lyrics
    encoded = [torch.tensor(encode_lyrics(l), dtype=torch.long) for l in lyrics_list]
    target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    targets = torch.cat(encoded, dim=0)

    # CTC input lengths: we’re not downsampling in time, so use poly_lengths
    input_lengths = (poly_lengths + 7) // 8

    return poly_padded, voc_padded, targets, input_lengths, target_lengths, lyrics_list


# ───────────────────────────────────────────────────────────────
# Dual-stream CNN + BiLSTM encoder
# ───────────────────────────────────────────────────────────────

class DualCNNBiLSTM(nn.Module):
    def __init__(self, n_mfcc=40, num_classes=NUM_CLASSES, hidden_size=256, num_layers=2):
        super().__init__()

        # Poly branch
        self.poly_conv1 = nn.Conv1d(n_mfcc, 64, kernel_size=3, stride=2, padding=1)
        self.poly_conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)

        # Vocal branch
        self.voc_conv1 = nn.Conv1d(n_mfcc, 64, kernel_size=3, stride=2, padding=1)
        self.voc_conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)

        # Fusion: concat -> 256 channels
        self.fuse_conv = nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1)

        # BiLSTM over time
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, mfcc_poly, mfcc_voc):
        # mfcc_*: [B, n_mfcc, T]
        p = F.relu(self.poly_conv1(mfcc_poly))
        p = F.relu(self.poly_conv2(p))

        v = F.relu(self.voc_conv1(mfcc_voc))
        v = F.relu(self.voc_conv2(v))

        x = torch.cat([p, v], dim=1)      # [B, 256, T]
        x = F.relu(self.fuse_conv(x))     # [B, 256, T]

        x = x.permute(0, 2, 1)            # [B, T, 256]
        x, _ = self.rnn(x)                # [B, T, 2*hidden]
        x = self.fc(x)                    # [B, T, num_classes]
        x = x.permute(1, 0, 2)            # [T, B, num_classes]

        return F.log_softmax(x, dim=2)


# ───────────────────────────────────────────────────────────────
# Training loop with AMP
# ───────────────────────────────────────────────────────────────

def train(
    processed_folder: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    checkpoint_path: str = "dual_cnn_bilstm_ctc10.pt"
):
    epoch_count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INIT] Using device:", device)

    torch.backends.cudnn.benchmark = True

    dataset = LyricsDataset(processed_folder)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    model = DualCNNBiLSTM().to(device)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"[LOAD] Loading model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(epochs):
        if epoch_count + 1 % 5 == 0:
            torch.save(model.state_dict(), "dual_cnn_bilstm_ctcn"+str(epoch_count)+".pt")
        model.train()
        total_loss = 0.0

        for mfcc_poly, mfcc_voc, targets, input_lengths, target_lengths, raw_text in loader:
            mfcc_poly = mfcc_poly.to(device, non_blocking=True)
            mfcc_voc = mfcc_voc.to(device, non_blocking=True)
            targets = targets.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device.type):
                log_probs = model(mfcc_poly, mfcc_voc)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[EPOCH {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)

        # quick sanity decode
        if (epoch + 1) % 5 == 0 and len(dataset) > 0:
            model.eval()
            with torch.no_grad():
                sample_poly, sample_voc, sample_lyrics = dataset[0]
                sample_poly = sample_poly.unsqueeze(0).to(device)
                sample_voc = sample_voc.unsqueeze(0).to(device)

                out = model(sample_poly, sample_voc)  # [T, 1, C]
                pred_indices = out.argmax(dim=2)[:, 0].cpu().tolist()
                predicted = decode_output(pred_indices) # greedy decode for quick check during training
                
                print("  [SAMPLE]")
                print("   Target:   ", sample_lyrics[:80])
                print("   Predicted:", predicted[:80])
        epoch_count += 1

    torch.save(model.state_dict(), "dual_cnn_bilstm_ctcn120.pt")
    print("[DONE] Saved model to dual_cnn_bilstm_ctc120.pt")


if __name__ == "__main__":
    train("DALI/DALI_v1.0/processed-training/metadata", epochs=EPOCHS, batch_size=8, lr=1e-5, checkpoint_path="dual_cnn_bilstm_ctc120.pt")
