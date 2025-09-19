#comb_csv.csv


import pandas as pd
import json
from collections import Counter

csv_path = 'extracted_data.csv'

#transcription_column = 'original_text_no_punc' # can change to other columns later for original and no period/comma
transcription_column = 'clean_text'


"""

"""


df = pd.read_csv(csv_path)

def remove_punc(row):
    #print(row)
    #print(row['original_text_no_punc'])
    return row['text'].lower().replace('=', '').replace('#', '').replace('/', '').replace('(', '').replace(')', '').replace('\'', '').replace('*', '').replace('+', '').replace('?', '').replace('\"', '').replace('!', '').replace(',', '').replace('-', '').replace('.', '').replace(':', '')


# 2, '!': 3, ',': 4, '-': 5, '.': 6, ':'

df['clean_text'] = df.apply(remove_punc, axis=1)
#df.to_csv('full_segments.csv', index=False)

all_text = " ".join(df[transcription_column].astype(str).tolist()) # concatenate all transcriptions
char_counts = Counter(all_text)
vocab_list = sorted(char_counts.keys())

# special tokens
special_tokens = {"<pad>": 0, "<unk>": 1} # Standard practice

vocab_dict = special_tokens.copy()
idx = len(special_tokens)
for char in vocab_list:
    if char not in vocab_dict: # don't overwrite special tokens if they appear in text
        vocab_dict[char] = idx
        idx += 1

# save vocabulary
vocab_json_path = "./vocab.json"
with open(vocab_json_path, 'w', encoding='utf-8') as vocab_file:
    json.dump(vocab_dict, vocab_file, ensure_ascii=False, indent=4) # ensure_ascii=False for IPA

print(f"Vocabulary created with {len(vocab_dict)} tokens: {vocab_json_path}")
print(vocab_dict)

