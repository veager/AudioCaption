import re
import os
import torch
import torchaudio
import glob
import shutil
import moviepy.editor as mpy

torch.set_num_threads(1)



def Secs2Time(secs):
    # secs: seconds
    sec = int(secs) # 整数部分
    ms = round(1000 * (secs - int(secs)))  # 小数部分
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    time_str = '{0:0>2d}:{1:0>2d}:{2:0>2d}.{3:0>3d}'.format(h, m, s, ms)
    return time_str
# -----------------------------------------------------------


def ExtractAudio(video_path):
        '''
        from video extra the audio

        :param video_path: str
        :param audio_path: str
        :return:
        '''
        # video name with suffix
        video_suffix = '.' + video_path.split('.')[-1]
        # audio path
        audio_path = video_path.replace(video_suffix, '.wav')
        # read video, and extract the audio
        audio = mpy.VideoFileClip(video_path).audio
        # save audio file
        audio.write_audiofile(audio_path)

        return None
# -----------------------------------------------------------


def AudioVAD(audio_path, save_folder):
    '''

    :param audio_path:
    :param save_folder:
    :return:
    '''
    sampling_rate = 16000

    # read model
    model, utils = torch.hub.load(
        repo_or_dir = 'snakers4/silero-vad',
        model = 'silero_vad',
        force_reload = True
    )

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # load audio
    audio = read_audio(audio_path, sampling_rate=sampling_rate)
    # print(wav.size(dim=0), 'duration:', wav.size(dim=0) / SAMPLING_RATE)
    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=sampling_rate)

    timer = []
    for i, timestamp in enumerate(speech_timestamps):

        timer.append({
            'no' : i+1,
            'duration' : (timestamp['end'] - timestamp['start']) / sampling_rate,   # audio duration, unit: sec
            'start' : timestamp['start'] / sampling_rate,
            'end' : timestamp['end'] / sampling_rate
        })

        # audio file name with file suffix
        audio_name = os.path.basename(audio_path)
        save_name = audio_name.replace('.wav', '_{0:0>6d}.wav'.format(i+1))
        save_path = os.path.join(save_folder, save_name)
        save_audio(save_path, collect_chunks([timestamp], audio), sampling_rate=sampling_rate)

    return timer
# -----------------------------------------------------------


def AudioSTT(audio_folder):
    '''

    :param audio_folder:
    :return:
    '''
    device = torch.device('cpu')
    # gpu also works, but our models are fast enough for CPU
    model, decoder, utils = torch.hub.load(
        repo_or_dir = 'snakers4/silero-models',
        model = 'silero_stt',
        language = 'en',
        device = device)

    (read_batch, split_into_batches, read_audio, prepare_model_input) = utils  # see function signature for details

    test_files = [os.path.join(audio_folder, p) for p in os.listdir(audio_folder)]
    test_files = sorted(test_files, key=lambda x: int(re.findall('\d+', x.split('\\')[-1].split('.')[0])[0]))

    batch_size = 10
    batches = split_into_batches(test_files, batch_size=batch_size)

    text_r = []
    for j in range(len(batches)):

        input = prepare_model_input(read_batch(batches[j]), device=device)
        # prediction
        output = model(input)
        for i, example in enumerate(output):
            text = decoder(example.cpu())
            text_r.append({'no': j*batch_size+i+1, 'text': text})
    return text_r
# -----------------------------------------------------------

def CreatText(text, path):
    with open(path, 'w') as f:
        for tt in text:
            content = tt['text'].strip()
            if len(content) > 0:
                f.writelines(content)
                f.writelines(' ')
    return None
# -----------------------------------------------------------

def CreatCaption(timer, text, path, front_extend=0.2, behind_extend=0.2):
    '''

    :param timer:
    :param text:
    :param path:
    :param front_extend
    :param behind_extend

    :return:
    '''

    with open(path, 'w') as f:
        j = 0
        for tm, tt in zip(timer, text):
            assert tt['no'] == tm['no']

            content = tt['text'].strip()
            if len(content) > 0:
                j = j + 1
                f.writelines(str(j))
                f.writelines('\n')
                f.writelines(Secs2Time(tm['start']-front_extend) + ' --> ' + Secs2Time(tm['end']+behind_extend))
                f.writelines('\n')
                f.writelines(content)
                f.writelines('\n')
                f.writelines('\n')

    return None
# -----------------------------------------------------------


def main():
    audio_path = 'data\\test.mp4'
    audio_path = os.path.join(os.getcwd(), audio_path)
    caption_path = None
    as_srt = True



    del_audio = False
    audio_suffix = '.' + audio_path.split('.')[-1]

    # caption path
    if caption_path is None:
        caption_path = audio_path.replace(audio_suffix, '.srt')

    # create __temp__ folder
    temp_folder = os.path.join(os.getcwd(), '__temp__')
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    if not (audio_suffix in ['.wav']):
        # will delete the extracted audio later
        del_audio = True
        # extract audio
        ExtractAudio(audio_path)
        audio_path = audio_path.replace(audio_suffix, '.wav')

    #
    timer = AudioVAD(audio_path, temp_folder)
    #
    text = AudioSTT(temp_folder)
    # save caption file
    if as_srt:
        CreatCaption(timer, text, caption_path)
    else:
        CreatText(text, caption_path)

    # delete the temporary files
    shutil.rmtree(temp_folder)
    if del_audio:
        os.remove(audio_path)
# -----------------------------------------------------------


if __name__ == "__main__":
    main()



