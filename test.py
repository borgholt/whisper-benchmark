import os
from time import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from rich import print
from tqdm import tqdm
from faster_whisper import WhisperModel
from pytube import YouTube
from pydub import AudioSegment

# audio loading parameters
TEST_AUDIO_URL = "https://www.youtube.com/watch?v=0u7tTptBo9I"
FRAME_RATE = 16_000
SAMPLE_WIDTH = 2
CHANNELS = 1

# model loading parameters
MODEL_PATH_OR_SIZE = "large-v2"
NUM_WORKERS = 1
COMPUTE_TYPE = "float16"
DEVICE = "cuda"

# inference parameters
BEAM_SIZE = 5
LANGUAGE = "fr"
CONDITION_ON_PREVIOUS_TEXT = False
VAD_FILTER = False
WORD_TIMESTAMPS = False


def get_youtube_audio(url: str) -> str:
    os.makedirs("tmp", exist_ok=True)
    yt = YouTube(url)
    audio = yt.streams.filter(only_audio=True).first()
    filepath = audio.download(output_path="tmp", skip_existing=True)
    return filepath


def evaluation(segments, info):
    audio_duration = info.duration
    duration_progress = round(audio_duration, 2)
    with tqdm(total=duration_progress, unit=" s") as pbar:
        total_update = 0
        for segment in segments:
            segment_update = segment.end - total_update
            segment_update = max(
                min(segment_update, duration_progress - total_update), 0
            )
            segment_update = round(segment_update, 2)
            total_update += segment_update
            pbar.update(segment_update)
        process_duration = pbar.format_dict["elapsed"]
        last_update = max(duration_progress - total_update, 0)
        pbar.update(last_update)
    irtf = audio_duration / process_duration
    return process_duration, audio_duration, irtf


# load audio
filepath = get_youtube_audio(TEST_AUDIO_URL)
audio_segment = AudioSegment.from_file(filepath)
audio_segment = audio_segment.set_frame_rate(FRAME_RATE)
audio_segment = audio_segment.set_sample_width(SAMPLE_WIDTH)
audio_segment = audio_segment.set_channels(CHANNELS)
audio = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)

# load model
model = WhisperModel(
    MODEL_PATH_OR_SIZE,
    compute_type=COMPUTE_TYPE,
    num_workers=NUM_WORKERS,
    device=DEVICE,
)

# run inference
segments, info = model.transcribe(
    audio,
    vad_filter=VAD_FILTER,
    language=LANGUAGE,
    beam_size=BEAM_SIZE,
    condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
    word_timestamps=WORD_TIMESTAMPS,
)
process_duration, audio_duration, irtf = evaluation(segments, info)

print("\nAudioSegment.from_file( ... ).set_something( ... )")
print(f"URL:\t\t\t{TEST_AUDIO_URL}")
print(f"Duration:\t\t{audio_duration / 60:.2f} minutes")

print("\nWhisperModel( ... )")
print(f"Model:\t\t\t{MODEL_PATH_OR_SIZE}")
print(f"Device:\t\t\t{DEVICE}")
print(f"Workers:\t\t{NUM_WORKERS}")
print(f"Compute type:\t\t{COMPUTE_TYPE}")

print("\nmodel.transcrib( ... )")
print(f"Beam size:\t\t{BEAM_SIZE}")
print(f"Prev. text:\t\t{CONDITION_ON_PREVIOUS_TEXT}")
print(f"Language:\t\t{LANGUAGE}")
print(f"VAD filter:\t\t{VAD_FILTER}")
print(f"Word timestamps:\t{WORD_TIMESTAMPS}")

print("\n[bold purple]Results[/bold purple]")
print(f"Inference duration:\t[bold green]{process_duration:.2f}[/bold green] seconds")
print(f"Inverse RTF:\t\t[bold green]{irtf:.2f}[/bold green]\n")
