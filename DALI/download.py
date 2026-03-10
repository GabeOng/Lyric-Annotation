import os
import yt_dlp
import DALI as dali

# -------- CONFIG --------

ANNOTATION_PATH = "DALI_v2.0/annotations"
OUTPUT_AUDIO = "audio"

MAX_DOWNLOADS = 50   # change later

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

        ydl.download([url])

        count += 1

    except Exception as e:

        print("Failed:", song_id, e)

print("Downloaded:", count)