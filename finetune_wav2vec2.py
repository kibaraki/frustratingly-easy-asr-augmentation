import os
import pandas as pd
import torch
import torchaudio
from datasets import load_dataset, Audio, DatasetDict
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import evaluate 
import json
import jiwer
from argparse import ArgumentParser
import argparse
import random

import torch.nn as nn
import torch.nn.functional as F

import logging

parser = ArgumentParser(description="shibur asr")
parser.add_argument(
    "--epoch",
    type=int,
    default=30,
    help="Number of epochs"
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=3e-5,
    help="Learning rate for training"
)
parser.add_argument(
    "--output",
    type=str,
    default="./wav2vec2-finetuned-ipa",
    help="Output directory for saved model"
)
parser.add_argument(
    "--pretrained",
    type=str,
    default= "facebook/wav2vec2-large-xlsr-53",
    help="Pretrained model to use"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size for training"
)
parser.add_argument(
    "--cer_wer",
    type=str,
    default="cer",
    choices=["cer", "wer", "cerwer"],
    help="Whether to use just CER as metric or CER & WER"
)
parser.add_argument(
    "--ignore_mismatch",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Use --no-ignore_mismatch if want to keep it false"
)
parser.add_argument(
    "--resampled",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Use --no-resampled if want to use original audio and resample during training"
)
parser.add_argument(
    "--train",
    type=str,
    default="split/shibur_train70_21_res.csv",
    help="Dataset to train on"
)
parser.add_argument(
    "--eval",
    type=str,
    default="split/shibur_val15_21_res.csv",
    help="Dataset to evaluate on"
)
parser.add_argument(
    "--vocab",
    type=str,
    default="./vocab.json",
    help="JSON file containing vocab"
)
parser.add_argument(
    "--tran_column",
    type=str,
    default= "original_text_no_punc",
    help="CSV column with desired transcription"
)
parser.add_argument(
    "--man_aug",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Use --no-man_aug if no data augmentation for training"
)
parser.add_argument(
    "--freeze",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Use --no-freeze if don't want to do model.freeze_feature_encoder(), for later iterations of fine-tuning"
)
parser.add_argument(
    "--cmat",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Use --no-cmat if don't want to use confusion matrix from clean_0_train"
)
parser.add_argument(
    "--conf_lambda",
    default=0.5,
    type=float,
    help="lambda value for confusion penalty, default 0.5"
)
parser.add_argument(
    "--conf_csv",
    default="confusion_matrix_train0.csv",
    type=str,
    help="CSV for confusion matrix"
)
parser.add_argument(
    "--spec_aug",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Use --no-spec_aug to turn off"
)
parser.add_argument(
    "--man_aug_prob",
    type=float,
    default=0.4,
    help="Probability for manual augmentation, default 0.4"
)
parser.add_argument(
    "--spec_aug_prob",
    type=float,
    default=0.1,
    help="Probability for spec augmentation, default 0.1"
)
parser.add_argument(
    "--spec_aug_len",
    type=int,
    default=10,
    help="Mask length for spec augmentation, for time and feature"
)

args = parser.parse_args()

print('=================================')
print('args passed:')
print(args)
print('=================================')


MODEL_ID = args.pretrained #"facebook/wav2vec2-large-xlsr-53"
OUTPUT_DIR = args.output #"./wav2vec2-finetuned-ipa"
VOCAB_JSON_PATH = args.vocab #"./vocab.json" 

# Need to resample audio anyways, but checking to make sure no difference between
# resampling during training, and using already resampled audio

# results seem comparable, keeping original for now

TRAIN_CSV_PATH = args.train #"shibur_train70_21.csv"     
VAL_CSV_PATH = args.eval #"shibur_val15_21.csv"  

CLIP_FILENAME_COL = "clip_filename"


# all *_res.csv files have "clip_filename" and "res_clip_filename", so if resampled, just change the column
if args.resampled:
    #TRAIN_CSV_PATH = "shibur_train70_21_res.csv"     
    #VAL_CSV_PATH = "shibur_val15_21_res.csv" 
    CLIP_FILENAME_COL = "res_clip_filename"
#############################################

print('====================================')
print(f'model: {MODEL_ID}')
print(f'Train: {TRAIN_CSV_PATH}')
print(f'Val:   {VAL_CSV_PATH}')
print('====================================')

#CLIP_FILENAME_COL = "clip_filename"

# Update if need to use transcription with punctuation
TRANSCRIPTION_COL = args.tran_column #"original_text_no_punc" 

NUM_EPOCHS = args.epoch #30 # check val loss 
LEARNING_RATE = args.learning_rate #3e-5 # 1e-4, 5e-5, 3e-5
BATCH_SIZE_TRAIN = args.batch_size #4
BATCH_SIZE_EVAL = args.batch_size #4
GRAD_ACCUMULATION_STEPS = 2 
#WARMUP_STEPS = 500
FP16 = torch.cuda.is_available() 

# load datasets
print("Loading datasets")
data_files = {
    "train": TRAIN_CSV_PATH,
    "validation": VAL_CSV_PATH
}
raw_datasets = load_dataset("csv", data_files=data_files, cache_dir="./cache")

print("Set up tokenizer and feature extractor")
# Tokenizer and Feature Extractor
# Load the custom vocabulary
tokenizer = Wav2Vec2CTCTokenizer(
    VOCAB_JSON_PATH,
    unk_token="<unk>",
    pad_token="<pad>",
    word_delimiter_token=" " if " " in json.load(open(VOCAB_JSON_PATH)) else None, 
)

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True 
)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

###
print("Vocab size:", len(processor.tokenizer))
print("Pad / blank token index:", processor.tokenizer.pad_token_id)
print("Sample token → character:")
for idx in [processor.tokenizer.pad_token_id, 0, 1, 2, 3, 4]:
    print(f"  {idx} → '{processor.tokenizer.convert_ids_to_tokens(idx)}'")
###


print("--- Tokenizer Vocabulary ---")

# Get the vocabulary dictionary {token_string: token_id}
vocab_dict = processor.tokenizer.get_vocab()

sorted_vocab = sorted(vocab_dict.items(), key=lambda item: item[1])

# Print each token and its corresponding ID
for token, token_id in sorted_vocab:
    print(f"ID: {token_id:<5} Token: '{token}'")

print(f"\nTotal vocabulary size: {len(sorted_vocab)}")



################ ADDED for augmentation #########################
def manual_augment(waveform, sample_rate):
   
    orig = waveform

    # Speed Perturbation
    if random.random() < args.man_aug_prob: # 0.4: # changed to 50% chance to apply
        #speed_factor = random.choice([0.9, 1.1])
        print('speed perturb')
        speed_factor = random.uniform(0.9, 1.1) #(0.9, 1.1)
        try:
            waveform, _ = torchaudio.functional.speed(waveform, sample_rate, speed_factor)
        except Exception as e:
            print(f"Could not apply speed augmentation: {e}")
            pass
    
    # Pitch Shift
    if random.random() < args.man_aug_prob:  # 0.4: # changed to 30% chance to apply
        print('pitch shift')
        pitch_shift = random.randint(-3, 3) #(-1, 1)
        if pitch_shift != 0: 
            try:
                waveform = torchaudio.functional.pitch_shift(waveform, sample_rate, pitch_shift)
            except Exception as e:
                # print(f"Could not apply pitch augmentation: {e}")
                pass

    # Add Gaussian Noise
    if random.random() < args.man_aug_prob: # 0.4: # changed to 20% chance to apply
        print('gauss noise')
        noise_level = random.uniform(0.001, 0.01)    # all other runs using (0.001, 0.01)
        noise = torch.randn_like(waveform) * noise_level
        waveform = waveform + noise
    
    # 1600 samples = 0.1 seconds at 16kHz, try 400
    MIN_LENGTH = 400 
    if waveform.shape[-1] < MIN_LENGTH:
        print(f"Warning: Augmented waveform is too short ({waveform.shape[-1]} samples). Skipping augmentation for this item.")
        return orig  # Or return the original, un-augmented waveform
    
    if not torch.isfinite(waveform).all():
        print("Warning: Augmentation produced non-finite values (inf or nan).")
        return orig  # Or return original waveform

    return waveform



# takes an 'apply_augmentation' flag and works with tensors
# before converting to numpy at the very end.
def speech_file_to_array_fn(batch, apply_augmentation=False):
    try:
        print(batch[CLIP_FILENAME_COL])

        speech_tensor, sampling_rate = torchaudio.load(batch[CLIP_FILENAME_COL])

        # Conditionally apply augmentations to the tensor
        # if apply_augmentation:
        #    print('-------------------aug-----------------------')
        #    speech_tensor = manual_augment(speech_tensor, processor.feature_extractor.sampling_rate)
 
        if sampling_rate != processor.feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampling_rate,
                new_freq=processor.feature_extractor.sampling_rate
            )
            speech_tensor = resampler(speech_tensor)
        
        # Ensure audio is mono by averaging channels if necessary
        if speech_tensor.shape[0] > 1:
            speech_tensor = torch.mean(speech_tensor, dim=0, keepdim=True)

        # Conditionally apply augmentations to the tensor
        if apply_augmentation:
            print("doing man aug")
            aug_tensor = manual_augment(speech_tensor, processor.feature_extractor.sampling_rate)
            
            if aug_tensor is not None:
                speech_tensor = aug_tensor
            else:
                print(f"aug failed for {batch[CLIP_FILENAME_COL]}")


        batch["speech"] = speech_tensor.squeeze().numpy()
        batch["sampling_rate"] = processor.feature_extractor.sampling_rate
        batch["target_text"] = str(batch[TRANSCRIPTION_COL])
        
    except Exception as e:
        print(f"Error loading or processing audio file {batch[CLIP_FILENAME_COL]}: {e}")
        batch["speech"] = None
        batch["target_text"] = ""
        
    return batch



print("Preprocessing audio data...")

if args.man_aug:
    print("Applying preprocessing and augmentation to the training set...")
    train_dataset = raw_datasets["train"].map(
        lambda batch: speech_file_to_array_fn(batch, apply_augmentation=True),
        num_proc=os.cpu_count() // 2 if os.cpu_count() > 2 else 1
    )   
else:
    print("Applying preprocessing to the training set (no augmentation) ...")
    train_dataset = raw_datasets["train"].map(
        lambda batch: speech_file_to_array_fn(batch, apply_augmentation=False),
        num_proc=os.cpu_count() // 2 if os.cpu_count() > 2 else 1
    )

print("Applying preprocessing to the validation set (no augmentation)...")
validation_dataset = raw_datasets["validation"].map(
    lambda batch: speech_file_to_array_fn(batch, apply_augmentation=False), # aug is False
    num_proc=os.cpu_count() // 2 if os.cpu_count() > 2 else 1
)

audio_loaded_datasets = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset
})


# calc warmup
num_rows = len(train_dataset)
steps_per_epoch = num_rows // args.batch_size 
total_steps = steps_per_epoch * args.epoch
warmup = int(total_steps * 0.1) # 10% for warmup

print(f'warmup steps: {warmup}')
print(f'total steps: {total_steps}')

WARMUP_STEPS = warmup




print("filter out failed audio loadings")
audio_loaded_datasets = audio_loaded_datasets.filter(lambda x: x["speech"] is not None)


def prepare_dataset(batch):
    # Process audio
    inputs = processor(batch["speech"], sampling_rate=batch["sampling_rate"], padding=True, return_attention_mask=True)
    batch["input_values"] = inputs.input_values[0]
    batch["attention_mask"] = inputs.attention_mask[0]

    # Process labels
    with processor.as_target_processor(): # set processor for text encoding
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch


print("Vectorizing dataset...")
processed_datasets = audio_loaded_datasets.map(
    prepare_dataset,
    remove_columns=audio_loaded_datasets["train"].column_names, # remove old columns
    batched=False, # process one by one for prepare_dataset as written
    num_proc=os.cpu_count() // 2 if os.cpu_count() > 2 else 1
)

print("\nFinal processed dataset:")
print(processed_datasets)
##################################################



# ORIGINAL #######################
"""
def speech_file_to_array_fn(batch, apply_augmentation=False):
    # speech_array, sampling_rate = torchaudio.load(os.path.join(BASE_AUDIO_PATH, batch[CLIP_FILENAME_COL]))
    try:
        speech_array, sampling_rate = torchaudio.load(batch[CLIP_FILENAME_COL])
        if sampling_rate != processor.feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=processor.feature_extractor.sampling_rate)
            batch["speech"] = resampler(speech_array).squeeze().numpy()
        else:
            batch["speech"] = speech_array.squeeze().numpy()

        # ensure audio is mono
        if len(batch["speech"].shape) > 1:
            batch["speech"] = batch["speech"][0] 


        batch["sampling_rate"] = processor.feature_extractor.sampling_rate
        
        batch["target_text"] = str(batch[TRANSCRIPTION_COL])
        #print(batch['target_text'])
    except Exception as e:
        print(f"Error loading or processing audio file {batch[CLIP_FILENAME_COL]}: {e}")
        batch["speech"] = None
        batch["target_text"] = ""
    return batch


print("Preprocessing audio data...")
raw_datasets = raw_datasets.map(
    speech_file_to_array_fn,
    num_proc=os.cpu_count() // 2 if os.cpu_count() > 2 else 1 
)



print("filter out failed audio loadings")
# Filter out failed audio loadings
raw_datasets = raw_datasets.filter(lambda x: x["speech"] is not None)


def prepare_dataset(batch):
    # Process audio
    inputs = processor(batch["speech"], sampling_rate=batch["sampling_rate"], padding=True, return_attention_mask=True)
    batch["input_values"] = inputs.input_values[0]
    batch["attention_mask"] = inputs.attention_mask[0]

    # Process labels
    with processor.as_target_processor(): # set processor for text encoding
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

print("Vectorizing dataset...")
processed_datasets = raw_datasets.map(
    prepare_dataset,
    remove_columns=raw_datasets["train"].column_names, # remove old columns
    batched=False, # process one by one for prepare_dataset as written
    num_proc=os.cpu_count() // 2 if os.cpu_count() > 2 else 1
)
"""


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


#cer_metric = evaluate.load("cer")
#wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    print(f'pred: {pred_str}')
    print(f'label: {label_str}')

    #cer = cer_metric.compute(predictions=pred_str, references=label_str)
    #wer = wer_metric.compute(predictions=pred_str, references=label_str)

    #return {"wer": wer, "cer": cer}

    cer = jiwer.cer(reference=label_str, hypothesis=pred_str)
    wer = jiwer.wer(reference=label_str, hypothesis=pred_str)

    ret = {"cer": cer}
    
    if args.cer_wer == 'wer':
        ret = {"wer": wer}
    elif args.cer_wer == 'cerwer':
        ret = {"wer": wer, "cer": cer}
    
    return ret

# blank token <pad> is index 0 in vocab


model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_ID,
    attention_dropout=0.1, 
    hidden_dropout=0.1,    
    feat_proj_dropout=0.01, #0.0, 
    layerdrop=0.1,
    apply_spec_augment=args.spec_aug,
    mask_time_prob= args.spec_aug_prob, #0.1, #0.05, 
    mask_time_length= args.spec_aug_len, #10, #def 10
    mask_feature_prob= args.spec_aug_prob, #0.1, #0.05, # ADDED #####################################################
    mask_feature_length=args.spec_aug_len, #10, #def 10
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer), 
    # if fine-tuning a model with a different head size, add ignore_mismatched
    ignore_mismatched_sizes=args.ignore_mismatch
)

# freeze_feature_extractor is deprecated

if args.freeze:
    model.freeze_feature_encoder()



###
print("Model’s CTC‐blank index:", model.config.pad_token_id)
###


# comparing at 30 epochs, no difference, so keep using cer and wer, not eval_cer, eval_wer
#### TESTING eval_cer again

metr = args.cer_wer

if metr == "cerwer" or metr == "cer":
    metr = "eval_cer"
elif metr == "wer":
    metr = "eval_wer"

#if metr == "cerwer":
#    metr = "cer"

print(f'metric: {metr}')

train_steps = steps_per_epoch // 2
#print(f"# of train steps: {train_steps}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    group_by_length=True, 
    per_device_train_batch_size=BATCH_SIZE_TRAIN,
    #per_device_eval_batch_size=BATCH_SIZE_EVAL,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    eval_strategy="steps", #"epoch",
    eval_steps=100, #100,
    save_strategy="steps", #"epoch", #if epoch remove save_steps and eval_steps
    save_steps=100,
    num_train_epochs=NUM_EPOCHS,
    fp16= False, #FP16, # set to False to try to fix grad vanishing and exploding
    # max_grad_norm=1.0, # clip: try to fix grad exploding to nan
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    load_best_model_at_end=True, 
    metric_for_best_model=metr, #"eval_cer",  
    greater_is_better=False, # comment out?
    logging_dir=f"{OUTPUT_DIR}/runs",
    logging_steps=10, # log every 10 steps
    save_total_limit=2,
)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,
    early_stopping_threshold=0.001
)



trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"],
    tokenizer=processor.feature_extractor,
    # callbacks=[early_stopping_callback], ############## trying early stopping
)


print("Starting fine-tuning...")
trainer.train()

# remove?
print("Evaluate")
trainer.evaluate()

# print('no evaluate')
#

print("Saving fine-tuned model and processor...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"Fine-tuning complete. Model saved to {OUTPUT_DIR}")


