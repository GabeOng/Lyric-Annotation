import os
import yt_dlp
import DALI as dali
import time
from itertools import islice
# CONFIG

BASE = os.path.join(os.getcwd(), "DALI/DALI_v1.0/")
ANNOTATION_PATH = os.path.join(BASE, "annotations")
INFO_PATH = os.path.join(BASE, "info/DALI_DATA_INFO.gz")

BASE_OUTPUT = os.path.join(os.getcwd(), "DALI/DALI_v1.0/")
OUTPUT_AUDIO = os.path.join(BASE_OUTPUT, "audio")

MAX_DOWNLOADS = 1000

os.makedirs(OUTPUT_AUDIO, exist_ok=True)


# LOAD DATASET

dataset = dali.get_the_DALI_dataset(ANNOTATION_PATH)

print("Songs available:", len(dataset))


# YOUTUBE DOWNLOADER

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


# DOWNLOAD LOOP

count = 0

for song_id in islice(dataset, 3193, 4000):

    if count >= MAX_DOWNLOADS:
        break

    entry = dataset[song_id]

    try:
        url = entry.info["audio"]["url"]

        if url is None:
            continue

        # Extract YouTube video ID from URL and check if already downloaded
        video_id = ydl.extract_info(url, download=False).get("id")
        output_path = os.path.join(OUTPUT_AUDIO, f"{video_id}.wav")

        if os.path.exists(output_path):
            print(f"Skipping (already exists): {entry.info['title']}")
            count += 1
            continue

        print(f"Downloading: {entry.info['title']}")
        print(f"{count} / {MAX_DOWNLOADS}")
        ydl.download([url])

        count += 1
        #time.sleep(8)
    except Exception as e:
        print("Failed:", song_id, e)

print("Downloaded:", count)