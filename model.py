
from torch.utils.data import Dataset
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# LOAD DATATESET
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

        mfcc_poly = data["mfcc_poly"]
        mfcc_voc = data["mfcc_voc"]
        lyrics = data.get("lyrics", None)

        return mfcc_poly, mfcc_voc, lyrics

# CNN MODEL
class CNN(nn.Module):
    def __init__(self, n_mfcc=40, num_classes=30):
        """
        n_mfcc: number of MFCC features per frame
        num_classes: number of output tokens/characters
        """
        super(CNN, self).__init__()
        # Polyphonic branch
        self.conv1_poly = nn.Conv1d(in_channels=n_mfcc, out_channels=64, kernel_size=3, padding=1)
        self.conv2_poly = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Vocal Branch
        self.conv1_voc = nn.Conv1d(in_channels=n_mfcc, out_channels=64, kernel_size=3, padding=1)
        self.conv2_voc = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Conv layers
        self.conv1 = nn.Conv1d(in_channels=n_mfcc, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        x: [batch, n_mfcc, time_steps]
        """
        # --- Poly branch ---
        p = F.relu(self.poly_conv1(mfcc_poly))
        p = F.relu(self.poly_conv2(p))

        # --- Vocal branch ---
        v = F.relu(self.voc_conv1(mfcc_voc))
        v = F.relu(self.voc_conv2(v))

        # --- Fuse ---
        x = torch.cat([p, v], dim=1)   # [B, 256, T]
        x = F.relu(self.fuse_conv(x))  

        # transpose for fully connected layer
        x = x.permute(0, 2, 1)  # [batch, time_steps, features]

        x = self.fc(x)  # [batch, time_steps, num_classes]

        return F.log_softmax(x, dim=2)

if __name__ == "__main__":
    batch_size = 2
    n_mfcc = 40
    time_steps = 100
    num_classes = 30

    mfcc_poly = torch.randn(B, n_mfcc, T)
    mfcc_voc  = torch.randn(B, n_mfcc, T)
    #dummy_input = torch.randn(batch_size, n_mfcc, time_steps)

    # initialize model
    model = CNN(n_mfcc=n_mfcc, num_classes=num_classes)

    # forward pass
    output = model(mfcc_poly, mfcc_voc)

    print("Output shape:", output.shape)
    # should be [batch_size, time_steps, num_classes]
