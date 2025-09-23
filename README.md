# frustratingly-easy-asr-augmentation
Repository for FRUSTRATINGLY EASY DATA AUGMENTATION FOR LOW-RESOURCE ASR

## Paper
[arXiv](https://arxiv.org/abs/2509.15373)

## TL;DR
Data augmentation for ASR in low-resource languages, using only the training data.

## Data
- [Vatlongos](https://pangloss.cnrs.fr/corpus/Vatlongos?lang=en) | [(CC BY-NC-ND 3.0)](https://creativecommons.org/licenses/by-nc-nd/3.0/)
- [Nashta](https://pangloss.cnrs.fr/corpus/Nashta?lang=en) | [(CC BY-NC 2.5)](https://creativecommons.org/licenses/by-nc/2.5/)
- [Shinekhen Buryat](https://tufs.repo.nii.ac.jp/search?page=1&size=50&sort=custom_sort&search_type=2&q=1729497608274) | [(CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/deed.en), [Hugging Face](https://huggingface.co/datasets/kibaraki/Shinekhen-Buryat)
- [Kakabe](https://pangloss.cnrs.fr/corpus/Kakabe?lang=en) | [(CC BY-NC-ND 3.0)](https://creativecommons.org/licenses/by-nc-nd/3.0/)
  
## Code
- `finetune_wav2vec2.py`: fine-tuning the models
- `pipeline.py`: evaluating the models
- `data_processing/`
  - `adjust_cols.py`: adjust columns in TTS-generated csv to columns matching training data csv 
  - `combine_splits.py`: combine different splits into one csv
  - `create_vocab.py`: given all annotated data (before making splits), generate `vocab.json` of all phonemes/characters
  - `extract_crop_kakabe.py`: example of `extract_crop.py` to extract transcription from XML file
  - `sample.py`: sample the TTS-generated data, to maintain 1:1 ratio of synthetic to original data
  - `train_split.py`: given all annotated data, make train/val/test splits
- `data_augmentation/`
  - `gen_sentences.py`: generate synthetic sentences from training sentences, by swapping out words with alternatives, based on gloss and POS
  - `gen_random.py`: generate synthetic sentences from training sentences, by swapping out words randomly with words in vocabulary
- `analysis/`
  - `count_words.py`: count the number of unique words and glosses
  - `hal_count-new.py`: count the number of shared words (overlapping vocab) for two splits


## `finetune_wav2vec2.py`
```
python3 finetune_wav2vec2.py --epoch 30 --learning_rate 1e-4 --batch_size 4 \
  --cer_wer cer --output models/fl_e30_b4_lr1e-4_cer_random873+shib  --no-ignore_mismatch \
  --resampled --pretrained facebook/wav2vec2-large-xlsr-53  --train split/kokoro/random_gem873+shib.csv \
  --eval split/shibur_val15_0_res_clean.csv --tran_column clean_text --vocab vocab.json --no-man_aug \
  --freeze --no-spec_aug --man_aug_prob 0.05 --spec_aug_prob 0.3 --spec_aug_len 5
```

## `pipeline.py`
```
python3 pipeline.py --model models/fl_e30_b4_lr1e-4_cer_random873+shib --no-shibur \
  --dataset_1 split/shibur_val15_0_res_clean.csv --dataset_2 split/shibur_test15_0_res_clean.csv \
  --tran_col clean_text
```
## Augmentation Methods
- Replacing words by words with the same gloss
- Replacing words with random words
- Generating new sentences using a LLM (not included in repository)
