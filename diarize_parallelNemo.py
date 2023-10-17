import argparse
import os
from helpersNemo import *
import whisperx
import torch
from deepmultilingualpunctuation import PunctuationModel
import re
import subprocess
import logging


def run_nemo(whisper_model,args):
    ROOT = os.path.join("/home","ubuntu",".tmpaud")
    temp_path = os.path.join(ROOT, "temp_outputs")
    if args.stemming:
        # Isolate vocals from the rest of the audio

        return_code = os.system(
            f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o "temp_outputs"'
        )

        if return_code != 0:
            logging.warning(
                "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
            )
            vocal_target = args.audio
        else:
            vocal_target = os.path.join(
                temp_path, "htdemucs", os.path.basename(args.audio[:-4]), "vocals.wav"
            )
    else:
        vocal_target = args.audio

    logging.info("Starting Nemo process with vocal_target: ", vocal_target)
    nemo_process = subprocess.Popen(
        ["python3", "nemo_process.py", "-a", vocal_target, "--device", args.device],
        #stdout=subprocess.PIPE,
        #stderr=subprocess.PIPE,
    )


    segments, info = whisper_model.transcribe(
        vocal_target, beam_size=1, word_timestamps=True
    )
    whisper_results = []
    for segment in segments:
        whisper_results.append(segment._asdict())

    # clear gpu vram
    del whisper_model
    torch.cuda.empty_cache()
    print("language")
    print(info.language)

    if info.language in wav2vec2_langs:
        print("is into if")
        alignment_model, metadata = whisperx.load_align_model(
            language_code=info.language, device=args.device
        )
        result_aligned = whisperx.align(
            whisper_results, alignment_model, metadata, vocal_target, args.device
        )
        word_timestamps = result_aligned["word_segments"]
        # clear gpu vram
        del alignment_model
        torch.cuda.empty_cache()
    else:
        word_timestamps = []
        for segment in whisper_results:
            for word in segment["words"]:
                word_timestamps.append({"text": word[2], "start": word[0], "end": word[1]})

    # Reading timestamps <> Speaker Labels mapping
    nemo_process.communicate()
    

    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if info.language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")

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
        logging.warning(
            f'Punctuation restoration is not available for {info.language} language.'
        )

    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    with open(f"{args.audio[:-4]}.txt", "w", encoding="utf-8-sig") as f:
        print("this is the file")
        print(f"saved on {args.audio[:-4]}.txt")
        print(f)
        get_speaker_aware_transcript(ssm, f)

    with open(f"{args.audio[:-4]}.srt", "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    with open(f"{args.audio[:-4]}.json", "w", encoding="utf-8") as json_file:
        write_json(ssm, json_file)

    cleanup(temp_path)