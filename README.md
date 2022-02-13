# AudioCaption
This is a tool which can add captions for audio /  video. It is built in Python language based on the [Voice Activity Detector (VAD)](https://github.com/snakers4/silero-vad) model and [ Speech-To-Text (STT)](https://github.com/snakers4/silero-models) model of Silero.

## 1. Requirements

- pytorch
- torchaudio
- moviepy

## 2. Parameters

Three parameters need to be set in the first three lines of the `main` function.

- `audio_path`: audio path or video path; **avoid use any number in the audio / movie file name**
- `caption_path`: the directory to save caption file
- `as_srt`: to indicate the caption format:
  - `as_srt = True`: `.srt` format
  - `as_srt = False`: plain text

Your can also finely adjust the dwelling time of each caption by using the parameters `front_extend` and `behind_extend` of the function `CreatCaption`.
