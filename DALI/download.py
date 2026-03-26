import os
import yt_dlp
import DALI as dali

# -------- CONFIG --------

BASE = os.path.join(os.getcwd(), "DALI/DALI_v1.0/")   # adjust if needed

## MUST DOWNLOAD DALI DATASET FIRST AND EXTRACT TO DALI/DALI_v1.0/
ANNOTATION_PATH = os.path.join(BASE, "annotations")
INFO_PATH = os.path.join(BASE, "info/DALI_DATA_INFO.gz")

BASE_OUTPUT = os.path.join(os.getcwd(), "DALI/DALI_v1.0/")   # adjust if needed
OUTPUT_AUDIO = os.path.join(BASE_OUTPUT, "audio")   # adjust if needed


MAX_DOWNLOADS = 150   # change later

os.makedirs(OUTPUT_AUDIO, exist_ok=True)


# -------- LOAD DATASET --------

dataset = dali.get_the_DALI_dataset(ANNOTATION_PATH)

print("Songs available:", len(dataset))


# -------- YOUTUBE DOWNLOADER --------

ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": os.path.join(OUTPUT_AUDIO, "%(id)s.%(ext)s"),
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }
    ],
    "quiet": True
}

ydl = yt_dlp.YoutubeDL(ydl_opts)


# -------- DOWNLOAD LOOP --------

count = 0

for song_id in dataset:

    if count >= MAX_DOWNLOADS:
        break

    entry = dataset[song_id]

    try:

        url = entry.info["audio"]["url"]

        if url is None:
            continue

        print("Downloading:", entry.info["title"])
        print(count, "/", MAX_DOWNLOADS)
        ydl.download([url])

        count += 1

    except Exception as e:

        print("Failed:", song_id, e)

print("Downloaded:", count)