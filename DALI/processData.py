import os
import torch
import torchaudio
import torchaudio.transforms as T
import DALI as dali

ANNOTATION_PATH = "DALI_v2.0/annotations"
INFO_PATH = "DALI_v2.0/info/DALI_DATA_INFO.gz"

AUDIO_FOLDER = "audio"
OUTPUT_FOLDER = "processed"

MAX_SONGS = 100   # limit for testing

os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


#LOAD DALI

print("Loading DALI annotations...")

dataset = dali.get_the_DALI_dataset(ANNOTATION_PATH)
info = dali.get_info(INFO_PATH)

print("Total songs:", len(dataset))


#DOWNLOAD AUDIO

print("Downloading audio (may take time)...")

errors = dali.get_audio(info, AUDIO_FOLDER)

print("Download errors:", errors)


#MFCC

mfcc_transform = T.MFCC(
    sample_rate=44100,
    n_mfcc=40
)


#PROCESS

processed_count = 0

for song_id in dataset:

    if processed_count >= MAX_SONGS:
        break

    entry = dataset[song_id]

    try:

        #AUDIO PATH

        audio_file = os.path.join(AUDIO_FOLDER, song_id + ".wav")

        if not os.path.exists(audio_file):
            continue

        waveform, sr = torchaudio.load(audio_file)

        #MFCC

        mfcc = mfcc_transform(waveform)
        mfcc = mfcc.squeeze(0)   # remove channel dim


        #EXTRACT LYRICS

        words = entry.annotations["annot"]["words"]

        lyrics_list = [w["text"] for w in words]

        lyrics = " ".join(lyrics_list)

        #SAVE

        save_data = {
            "mfcc": mfcc,
            "lyrics": lyrics,
            "song_id": song_id
        }

        save_path = os.path.join(OUTPUT_FOLDER, song_id + ".pt")

        torch.save(save_data, save_path)

        processed_count += 1

        print("Processed:", song_id)

    except Exception as e:

        print("Error processing", song_id, e)


print("Finished preprocessing.")
print("Total processed:", processed_count)