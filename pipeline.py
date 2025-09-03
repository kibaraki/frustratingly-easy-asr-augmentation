from transformers import pipeline
import torch
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from argparse import ArgumentParser
import jiwer
import argparse

parser = ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="models/full_e30_b4_lr1e-4_cer_1",
    help="Model to use for predictions"
)

parser.add_argument(
    "--dataset_1",
    type=str,
    default= '',
    help="Additional datasets to run the model on"
)

parser.add_argument(
    "--dataset_2",
    type=str,
    default= '',
    help="Additional datasets to run the model on"
)
parser.add_argument(
    "--dataset_3",
    type=str,
    default=""
)
parser.add_argument(
    "--shibur",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Use --no-shibur if you don't want to use the shibur train and val for evaluation"
)
#parser.add_argument(
#    "--more_datasets",
#    default=True,
#    action=argparse.BooleanOptionalAction,
#    help="Use --no-more_datasets if you don't want to run additional"
#)
parser.add_argument(
    "--tran_col",
    default='original_text_no_punc',
    help="To use no_punc column or one of the columns with punctuation left in"
)
args = parser.parse_args()

MODEL_PATH = args.model # "./e2_b1_lr1e-5" # "./wav2vec2-finetuned-ipa"  
AUDIO_FILE_TO_TRANSCRIBE = "shibur_100_clips/100_seg193.mp3"
RESAMPLED_FILE = 'shibur_100_clips/res_100_seg193.mp3'

"""
waveform, sample_rate = torchaudio.load(AUDIO_FILE_TO_TRANSCRIBE)
resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)

resampled = resampler(waveform)
torchaudio.save(RESAMPLED_FILE, resampled, 16000)
"""


device = 0 if torch.cuda.is_available() else -1 # 0 for GPU, -1 for CPU

try:
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=MODEL_PATH,
        device=device
    )
    print(f"Pipeline loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    exit()

# Transcribe an audio file 


metrics = []
#errors = []

def evaluate(val_or_test_csv):
    errors = []

    df = pd.read_csv(val_or_test_csv)

    mask = df[args.tran_col].notna() & (df[args.tran_col] != "")
    df = df[mask]

    # audio_files = ["path/to/audio1.wav", "path/to/audio2.wav"]
    audio_files = df['res_clip_filename'].tolist()
    actual = df[args.tran_col].tolist()

     

    # remove spaces from transcription since trained without
    # actual = [a.replace(" ", "") for a in actual]

    transcriptions = asr_pipeline(audio_files)
    trans = []

    #for audio_file, trans in zip(audio_files, transcriptions):
    for i in range(len(transcriptions)):
        trans.append(transcriptions[i]['text'])
        print(f"{audio_files[i]}\nactual:  {actual[i]}\npredict: {transcriptions[i]['text']}\n-------------------------------")

        sent_cer = jiwer.cer(reference=actual[i], hypothesis=transcriptions[i]['text'])
        if sent_cer > 0:
            ## ADD function to get the incorrect chars
            errors.append([audio_files[i], actual[i], transcriptions[i]['text']])

    print(actual)
    print(trans)

    cer = jiwer.cer(reference=actual, hypothesis=trans)
    wer = jiwer.wer(reference=actual, hypothesis=trans)
    #wil = jiwer.wil(reference=actual, hypothesis=trans)

    print(f"For {val_or_test_csv}")
    print(f"cer: {cer}, wer: {wer}\n")
    tmp = [val_or_test_csv, cer, wer]
    metrics.append(tmp)

    # check errors
    """
    print(f'Errors in {val_or_test_csv}:')
    for er in errors:
        print(f'{er[0]}\nactual: {er[1].replace(" ", "")}\npred: {er[2].replace(" ", "")}')
        # ADD ###################################
        # mechanism to store filename, and a counter for incorrect  
        
        print('-------------------------------------------------')
    """



if args.shibur:
    evaluate('split/shibur_train70_0_res_clean.csv')
    evaluate('split/shibur_val15_0_res_clean.csv')
    #evaluate('shibur_test15_21_res.csv')

if len(args.dataset_1) > 0:
    evaluate(args.dataset_1)

if len(args.dataset_2) > 0:
    evaluate(args.dataset_2)

if len(args.dataset_3) > 0:
    evaluate(args.dataset_3)

print('======================================================')

for me in metrics:
    print(f"Evaluating: {me[0]}")
    print(f"cer: {me[1]} , wer: {me[2]}")
    print('-------------------------------')


