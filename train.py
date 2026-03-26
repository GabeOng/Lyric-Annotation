import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Vocabulary ──────────────────────────────────────────────────────────────

CHARS = "abcdefghijklmnopqrstuvwxyz' "
BLANK = 0
char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = CTC blank
idx_to_char = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 for blank


def encode_lyrics(lyrics):
    lyrics = lyrics.lower()
    return [char_to_idx[c] for c in lyrics if c in char_to_idx]


def decode_output(indices):
    # collapse repeated tokens and remove blanks (greedy CTC decode)
    result = []
    prev = None
    for idx in indices:
        if idx != BLANK and idx != prev:
            result.append(idx_to_char.get(idx, ""))
        prev = idx
    return "".join(result)


# ── Dataset ──────────────────────────────────────────────────────────────────

class LyricsDataset(Dataset):
    def __init__(self, folder):
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".pt")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        mfcc = data["mfcc"].float()     # [n_mfcc, time_steps]
        lyrics = data["lyrics"].lower()
        return mfcc, lyrics


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch):
    mfccs, lyrics = zip(*batch)

    # pad MFCCs to longest in batch [batch, n_mfcc, time]
    mfcc_lengths = torch.tensor([m.shape[1] for m in mfccs])
    mfccs_padded = torch.nn.utils.rnn.pad_sequence(
        [m.T for m in mfccs], batch_first=True
    ).permute(0, 2, 1)

    # encode lyrics to integers
    encoded = [torch.tensor(encode_lyrics(l), dtype=torch.long) for l in lyrics]
    target_lengths = torch.tensor([len(e) for e in encoded])
    targets = torch.cat(encoded)  # CTC expects flat target tensor

    return mfccs_padded, targets, mfcc_lengths, target_lengths, lyrics


# ── Model ─────────────────────────────────────────────────────────────────────

class CNN(nn.Module):
    def __init__(self, n_mfcc=40, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(n_mfcc, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [batch, n_mfcc, time]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.permute(0, 2, 1)         # [batch, time, features]
        x = self.fc(x)                  # [batch, time, num_classes]
        x = x.permute(1, 0, 2)         # [time, batch, num_classes] — CTC expects this
        return F.log_softmax(x, dim=2)


# ── Training ──────────────────────────────────────────────────────────────────

def train(processed_folder, epochs=20, batch_size=2, lr=1e-3):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    #mps not available for ctc_loss   
    # elif torch.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    dataset = LyricsDataset(processed_folder)
    print("Songs loaded:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    # zero_infinity=True prevents NaN loss when input is shorter than target

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for mfccs, targets, mfcc_lengths, target_lengths, raw_lyrics in loader:
            mfccs = mfccs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            log_probs = model(mfccs)  # [time, batch, num_classes]

            loss = criterion(log_probs, targets, mfcc_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

        # Sample decode every 5 epochs to see how it's doing
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample_mfcc, sample_lyrics = dataset[0]
                sample_input = sample_mfcc.unsqueeze(0).to(device)
                out = model(sample_input)  # [time, 1, num_classes]
                pred_indices = out.argmax(dim=2).squeeze(1).cpu().tolist()
                predicted = decode_output(pred_indices)
                print(f"  Target:    {sample_lyrics[:80]}")
                print(f"  Predicted: {predicted[:80]}")

    # Save model
    torch.save(model.state_dict(), "model.pt")
    print("Model saved to model.pt")


if __name__ == "__main__":
    train("DALI/DALI_v1.0/processed", epochs=5, batch_size=2)