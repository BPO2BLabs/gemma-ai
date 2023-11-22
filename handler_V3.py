import runpod
import os
import time
import subprocess
import shutil

#sleep_time = int(os.environ.get('SLEEP_TIME', 3))

## load your model(s) into vram here

######## CODE SERVE######################################
import os
import shutil
import subprocess
import time
import boto3
from botocore.exceptions import ClientError
import glob
import argparse

def get_files_with_extensions(directory, extensions):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and any(f.endswith(ext) for ext in extensions)]



###TRANSCRIPTOR#######################################


###########Import and Helper###################

import os
import wget
from omegaconf import OmegaConf
import json
import shutil
from faster_whisper import WhisperModel
import whisperx
import torch
import librosa
import soundfile
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
import logging
import requests

"""# Helper Functions"""

punct_model_langs = [
    "en",
    "fr",
    "de",
    "es",
    "it",
    "nl",
    "pt",
    "bg",
    "pl",
    "cs",
    "sk",
    "sl",
]
wav2vec2_langs = [
    "en",
    "fr",
    "de",
    "es",
    "it",
    "nl",
    "pt",
    "ja",
    "zh",
    "uk",
    "pt",
    "ar",
    "ru",
    "pl",
    "hu",
    "fi",
    "fa",
    "el",
    "tr",
]


def create_config(output_dir):
    DOMAIN_TYPE = "telephonic"  # Can be meeting or telephonic based on domain type of the audio file
    CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
    CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
    MODEL_CONFIG = os.path.join(output_dir, CONFIG_FILE_NAME)
    if not os.path.exists(MODEL_CONFIG):
        MODEL_CONFIG = wget.download(CONFIG_URL, output_dir)

    config = OmegaConf.load(MODEL_CONFIG)

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    meta = {
        "audio_filepath": os.path.join(output_dir, "mono_file.wav"),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open(os.path.join(data_dir, "input_manifest.json"), "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")

    pretrained_vad = "vad_multilingual_marblenet"
    pretrained_speaker_model = "titanet_large"

    config.num_workers = 1  # Workaround for multiprocessing hanging with ipython issue

    config.diarizer.manifest_filepath = os.path.join(data_dir, "input_manifest.json")
    config.diarizer.out_dir = (
        output_dir  # Directory to store intermediate files and prediction outputs
    )

    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.oracle_vad = (
        False  # compute VAD provided with model_path to vad config
    )
    config.diarizer.clustering.parameters.oracle_num_speakers = False
    config.diarizer.clustering.parameters.max_num_speakers=2

    # Here, we use our in-house pretrained NeMo VAD model
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.5
    config.diarizer.vad.parameters.offset = 0.1
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.msdd_model.model_path = (
        "diar_msdd_telephonic"  # Telephonic speaker diarization model
    )

    return config


def get_word_ts_anchor(s, e, option="start"):
    if option == "end":
        return e
    elif option == "mid":
        return (s + e) / 2
    return s


def get_words_speaker_mapping(wrd_ts, spk_ts, word_anchor_option="start"):
    s, e, sp = spk_ts[0]
    wrd_pos, turn_idx = 0, 0
    wrd_spk_mapping = []
    for wrd_dict in wrd_ts:
        ws, we, wrd = (
            int(wrd_dict["start"] * 1000),
            int(wrd_dict["end"] * 1000),
            wrd_dict["text"],
        )
        wrd_pos = get_word_ts_anchor(ws, we, word_anchor_option)
        while wrd_pos > float(e):
            turn_idx += 1
            turn_idx = min(turn_idx, len(spk_ts) - 1)
            s, e, sp = spk_ts[turn_idx]
            if turn_idx == len(spk_ts) - 1:
                e = get_word_ts_anchor(ws, we, option="end")
        wrd_spk_mapping.append(
            {"word": wrd, "start_time": ws, "end_time": we, "speaker": sp}
        )
    return wrd_spk_mapping


sentence_ending_punctuations = ".?!"


def get_first_word_idx_of_sentence(word_idx, word_list, speaker_list, max_words):
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    left_idx = word_idx
    while (
        left_idx > 0
        and word_idx - left_idx < max_words
        and speaker_list[left_idx - 1] == speaker_list[left_idx]
        and not is_word_sentence_end(left_idx - 1)
    ):
        left_idx -= 1

    return left_idx if left_idx == 0 or is_word_sentence_end(left_idx - 1) else -1


def get_last_word_idx_of_sentence(word_idx, word_list, max_words):
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    right_idx = word_idx
    while (
        right_idx < len(word_list)
        and right_idx - word_idx < max_words
        and not is_word_sentence_end(right_idx)
    ):
        right_idx += 1

    return (
        right_idx
        if right_idx == len(word_list) - 1 or is_word_sentence_end(right_idx)
        else -1
    )


def get_realigned_ws_mapping_with_punctuation(
    word_speaker_mapping, max_words_in_sentence=50
):
    is_word_sentence_end = (
        lambda x: x >= 0
        and word_speaker_mapping[x]["word"][-1] in sentence_ending_punctuations
    )
    wsp_len = len(word_speaker_mapping)

    words_list, speaker_list = [], []
    for k, line_dict in enumerate(word_speaker_mapping):
        word, speaker = line_dict["word"], line_dict["speaker"]
        words_list.append(word)
        speaker_list.append(speaker)

    k = 0
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k]
        if (
            k < wsp_len - 1
            and speaker_list[k] != speaker_list[k + 1]
            and not is_word_sentence_end(k)
        ):
            left_idx = get_first_word_idx_of_sentence(
                k, words_list, speaker_list, max_words_in_sentence
            )
            right_idx = (
                get_last_word_idx_of_sentence(
                    k, words_list, max_words_in_sentence - k + left_idx - 1
                )
                if left_idx > -1
                else -1
            )
            if min(left_idx, right_idx) == -1:
                k += 1
                continue

            spk_labels = speaker_list[left_idx : right_idx + 1]
            mod_speaker = max(set(spk_labels), key=spk_labels.count)
            if spk_labels.count(mod_speaker) < len(spk_labels) // 2:
                k += 1
                continue

            speaker_list[left_idx : right_idx + 1] = [mod_speaker] * (
                right_idx - left_idx + 1
            )
            k = right_idx

        k += 1

    k, realigned_list = 0, []
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k].copy()
        line_dict["speaker"] = speaker_list[k]
        realigned_list.append(line_dict)
        k += 1

    return realigned_list


def get_sentences_speaker_mapping(word_speaker_mapping, spk_ts):
    s, e, spk = spk_ts[0]
    prev_spk = spk

    snts = []
    snt = {"speaker": f"Speaker {spk}", "start_time": s, "end_time": e, "text": ""}

    for wrd_dict in word_speaker_mapping:
        wrd, spk = wrd_dict["word"], wrd_dict["speaker"]
        s, e = wrd_dict["start_time"], wrd_dict["end_time"]
        if spk != prev_spk:
            snts.append(snt)
            snt = {
                "speaker": f"Speaker {spk}",
                "start_time": s,
                "end_time": e,
                "text": "",
            }
        else:
            snt["end_time"] = e
        snt["text"] += wrd + " "
        prev_spk = spk

    snts.append(snt)
    return snts


def get_speaker_aware_transcript(sentences_speaker_mapping, f):
    for sentence_dict in sentences_speaker_mapping:
        sp = sentence_dict["speaker"]
        text = sentence_dict["text"]
        f.write(f"\n\n{sp}: {text}")


def format_timestamp(
    milliseconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert milliseconds >= 0, "non-negative timestamp expected"

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def write_srt(transcript, file):
    """
    Write a transcript to a file in SRT format.

    """
    for i, segment in enumerate(transcript, start=1):
        # write srt lines
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start_time'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end_time'], always_include_hours=True, decimal_marker=',')}\n"
            f"{segment['speaker']}: {segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )

def write_json(transcript, file):
    """
    Write a transcript to a file in JSON format.

    """
    list_of_file = file.name.split("/")
    fileName = list_of_file[len(list_of_file)-1]
    print(
            "{\n"
            "\"transcriptSegments\":\n"
            "[\n",
            file=file,
            flush=True)
    for i, segment in enumerate(transcript, start=1):
        # write srt lines
        last = ',\n' if i != len(transcript) else ''
        print(
            "{\n"
            f" \"segmentId\": {i},\n"
            f"\"segment\": \"{format_timestamp(segment['start_time'], always_include_hours=True, decimal_marker=',')} - "
            f"{format_timestamp(segment['end_time'], always_include_hours=True, decimal_marker=',')}\",\n"
            f"\"speaker\": \"{segment['speaker']}\", \n"
            f"\"text\": \"{segment['text'].strip().replace('-->', '->')}\"\n"
            "}"
            f"{ last }",
            file=file,
            flush=True,
        )
    print(
            "],\n"
            f"\"name\": \"{fileName}\""
            "}",
            file=file,
            flush=True,
    )

url_backend = "https://back.connectup.cloud"
#url_backend = "https://stage-test.connectup.cloud"
def send_json_to_backend(transcript, token, file):
    list_of_file = file.split("/")
    fileNameJSON = list_of_file[len(list_of_file)-1]
    fileNameMP3 = fileNameJSON[:-5] + ".mp3"
    transcription = []
    for i, segment in enumerate(transcript, start=1):
        transcription.append(
            {
            "segmentId": i,
            "segment": f"{format_timestamp(segment['start_time'], always_include_hours=True, decimal_marker=',')} - "
            f"{format_timestamp(segment['end_time'], always_include_hours=True, decimal_marker=',')}",
            "speaker": segment['speaker'],
            "text": segment['text'].strip().replace('-->', '->')
            })
    data = {
        "Transcripts" : [
            {
                "OriginalTranscriptSegments": transcription,
                "UpdatedTranscriptSegments": transcription,
                "name": fileNameMP3
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {token}"}
    response = requests.post(f"{url_backend}/Transcript/SaveBulkTranscripts", json=data, headers=headers)
    print(headers)
    print("backend status response: ",response.status_code)



def cleanup(path: str):
    """path could either be relative or absolute."""
    # check if file or directory exists
    if os.path.isfile(path) or os.path.islink(path):
        # remove file
        os.remove(path)
    elif os.path.isdir(path):
        # remove directory and all its content
        shutil.rmtree(path)
    else:
        raise ValueError("Path {} is not a file or dir.".format(path))


###################Loading in VRAM#################

# (choose from 'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large')
whisper_model_name = 'medium.en'
whisper_model = WhisperModel(whisper_model_name, device="cuda", compute_type="float16")

ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)

msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to("cuda")

punct_model = PunctuationModel(model="kredor/punctuate-all")

def handler(event):
    print("pre event")
    print(event)

    # Extract folder_name from the event dictionary
    folder_name = event.get('input', {}).get('folder')
    token_user = event.get('input', {}).get('token_user')
    print("token_user:", token_user)
    if not folder_name:
        print("Folder name not provided in event.")
        return "Error"
    if not token_user:
        print("Token user not provided in event.")
        return "Error"

    print("IN PROGRESS:")
    # do the things
    
    #subprocess.run(["python", "codeServe_V3.py", '--audio_folder' ,folder_name, '--token_user', f'"{token_user}"'], check=True)

    ###########OPTIONS########################

    # Name of the audio file
    #audio_path = '13334_Gexa_npineros__FrontierMain__26__-1__+18179443510__2023-08-03_10-45-57.wav'
    # import sys
    # audio_path = sys.argv[1]
    # token_user = sys.argv[2]

    s3_path = f"s3://gemma-middle-storage/{folder_name}"

    subprocess.run(['aws', 's3', 'sync', s3_path, '.'], check=True)

    audio_files = get_files_with_extensions('.', ['.mp3', '.wav'])

    for audio_file in audio_files:
        audio_path = os.path.basename(audio_file)
        print("Processing file: " + audio_file)

    # Whether to enable music removal from speech, helps increase diarization quality but uses alot of ram
    enable_stemming = True



    ###########Separating music from speech using Demucs############
    if enable_stemming:
        # Isolate vocals from the rest of the audio

        return_code = os.system(
            f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "temp_outputs"'
        )

        if return_code != 0:
            logging.warning(
                "Source splitting failed, using original audio file."
            )
            vocal_target = audio_path
        else:
            vocal_target = os.path.join(
                "temp_outputs", "htdemucs", os.path.basename(audio_path[:-4]), "vocals.wav"
            )
    else:
        vocal_target = audio_path

    ############WHISPER########################
    # Run on GPU with FP16
    

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = whisper_model.transcribe(
        vocal_target, beam_size=1, word_timestamps=True
    )
    whisper_results = []
    for segment in segments:
        whisper_results.append(segment._asdict())
    # clear gpu vram
    # del whisper_model
    # torch.cuda.empty_cache()

    #################alignment#########################
    word_timestamps = []
    for segment in whisper_results:
        for word in segment["words"]:
            word_timestamps.append({"text": word[2], "start": word[0], "end": word[1]})

    """## Convert audio to mono for NeMo combatibility"""

    signal, sample_rate = librosa.load(vocal_target, sr=None)
    # ROOT = os.getcwd()
    # temp_path = os.path.join(ROOT, "temp_outputs")
    # os.makedirs(temp_path, exist_ok=True)
    soundfile.write(os.path.join(temp_path, "mono_file.wav"), signal, sample_rate, "PCM_24")

    ##### NEMO MODEL#################################
    # Initialize NeMo MSDD diarization model
    
    msdd_model.diarize()

    ######## Reading timestamps <> Speaker Labels mapping ############

    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")


    ##### REALIGNING SPEECH############


    if info.language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word



        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    else:
        print(
            f'Punctuation restoration is not available for {whisper_results["language"]} language.'
        )

    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)


    ### AUX OUTPUT FUNC########

    send_json_to_backend(ssm, token_user, audio_path)

    with open(f"{audio_path[:-4]}.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(f"{audio_path[:-4]}.srt", "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    with open(f"{audio_path[:-4]}.json", "w", encoding="utf-8-sig") as json:
        write_json(ssm, json)


    def list_all_files(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                print(os.path.join(root, file))
  
    contents = os.listdir('.')

    # Print each item in the directory
    print('files in appTrans')
    for item in contents:
        print(item)

    #cleanup(temp_path)
    contents = os.listdir(temp_path)

    # Print each item in the directory
    print('files in temp_path')
    for item in contents:
        print(item)

    list_all_files(temp_path)

    

    def delete_subdirectory_contents(path):
        # Check if the directory exists
        if os.path.exists(path) and os.path.isdir(path):
            # Remove the directory and its contents
            shutil.rmtree(path)
            # Optionally, recreate the empty directory
            os.makedirs(path)
        else:
            print(f"The directory {path} does not exist.")

    # Define the paths of the subdirectories
    dir1 = "temp_outputs/htdemucs"
    dir2 = "temp_outputs/speaker_outputs"
    #dir3= "temp_outputs/mono_file.wav"

    # Delete the contents of these directories
    delete_subdirectory_contents(dir1)
    delete_subdirectory_contents(dir2)
    #os.remove(dir3)

    ###DELETED####
    print("DELETED")

    def list_all_files(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                print(os.path.join(root, file))
  
    contents = os.listdir('.')

    # Print each item in the directory
    print('files in appTrans')
    for item in contents:
        print(item)

    #cleanup(temp_path)
    contents = os.listdir(temp_path)

    # Print each item in the directory
    print('files in temp_path')
    for item in contents:
        print(item)

    list_all_files(temp_path)

    subprocess.run(['aws', 's3', 'sync', '.', 's3://gemma-middle-storage/testOutput/','--exclude', '*', '--include', '*.txt',
                    '--include', '*.json', '--include', '*.srt'], check=True)

    return "Done"

runpod.serverless.start({
    "handler": handler
})
