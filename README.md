# AudioCaption
This is a tool which can add captions for audio /  video. It is built in Python language based on the [Voice Activity Detector (VAD)]() model and [ Speech-To-Text (STT)]() model of Silero.

## 1. Requirements

- pytorch
- torchaudio
- moviepy

## 2. Parameters

Three parameters need to be set in the first three line of the `main` function.

- `audio_path`: audio path or video path
- `caption_path`: the directory to save caption file
- `as_srt`: to indicate the caption format:
  - `as_srt = True`: `.srt` format
  - `as_srt = False`: plain text

