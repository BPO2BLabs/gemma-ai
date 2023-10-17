import subprocess
import runpod
import os
import json
import boto3
from faster_whisper import WhisperModel
import diarize_parallelNemo as dpn

s3 = boto3.client('s3')
BUCKET_NAME = os.getenv("BUCKET_NAME")
BUCKET_KEY = os.getenv("BUCKET_KEY")
DOWNLOAD_PATH = os.getenv("DOWNLOAD_PATH")

mtypes = {'cpu': 'int8', 'cuda': 'float16'}

args = {
    "stemming": True,
    "model_name": "large-v2",
    "device": "cuda"
}
whisper_model = WhisperModel(
    args.model_name, device=args.device, compute_type=mtypes[args.device])


def transcribe(event):
  input= event.input
  target_s3 = input["file"]
  DOWNLOAD_FILE_PATH = os.path.join(DOWNLOAD_PATH, target_s3)
  s3.download_file(BUCKET_NAME, BUCKET_KEY, DOWNLOAD_FILE_PATH)

  args["audio"] = DOWNLOAD_FILE_PATH

  dpn.run_nemo(whisper_model, args)
  ROOT = os.path.join("/home","ubuntu",".tmpaud",target_s3)

  with open(f"{ROOT[:-4]}.json", 'r') as file_load:
      content_as_string = file_load.read()
      data = json.loads(content_as_string)
  return data

runpod.serverless.start({"handler":transcribe})