# frustrating-asr-augmentation
Repository for FRUSTRATINGLY EASY DATA AUGMENTATION FOR LOW-RESOURCE ASR

## TL;DR
Data augmentation for low-resource languages, using only the training data.

## Data
- [Vatlongos](https://pangloss.cnrs.fr/corpus/Vatlongos?lang=en) | [(CC BY-NC-ND 3.0)](https://creativecommons.org/licenses/by-nc-nd/3.0/)
- [Nashta](https://pangloss.cnrs.fr/corpus/Nashta?lang=en) | [(CC BY-NC 2.5)](https://creativecommons.org/licenses/by-nc/2.5/)
- [Shinekhen Buryat](https://tufs.repo.nii.ac.jp/search?page=1&size=50&sort=custom_sort&search_type=2&q=1729497608274) | [(CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/deed.en)

## Code
- `finetune_wav2vec2.py`: fine-tuning the models
- `pipeline.py`: evaluating the models
- `gen_sentences.py`: generate synthetic sentences from training sentences, by swapping out words with alternatives, based on gloss and POS
- `gen_random.py`: generate synthetic sentences from training sentences, by swapping out words randomly with words in vocabulary
