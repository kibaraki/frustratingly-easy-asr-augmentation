import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('split/train70.csv')

# vocab
vocab = {}
for _, row in df.iterrows():
    if pd.isna(row['clean_text']) or pd.isna(row['gloss']):
        continue
    
    clean_words = str(row['clean_text']).split()
    #gloss_words = str(row['pos']).split()
    gloss_words = ['a'] * len(clean_words)
    
    # one-to-one mapping
    if len(clean_words) == len(gloss_words):
        for i in range(len(clean_words)):
            # Use the gloss as a key for the part-of-speech/meaning
            if gloss_words[i] not in vocab:
                vocab[gloss_words[i]] = []
            if clean_words[i] not in vocab[gloss_words[i]]:
                vocab[gloss_words[i]].append(clean_words[i])


new_sentences = []

num_sentences_to_generate = 4811  

for i in range(num_sentences_to_generate):
    # Pick a random sentence to use as a template
    template_row = df.iloc[np.random.randint(0, len(df))]
    
    if pd.isna(template_row['clean_text']) or pd.isna(template_row['gloss']):
        continue

    original_clean_words = str(template_row['clean_text']).split()
    #original_gloss_words = str(template_row['pos']).split()
    original_gloss_words = ['a'] * len(original_clean_words)

    if len(original_clean_words) != len(original_gloss_words):
        continue

    new_clean_words = []
    new_gloss_words = []
    
    # Substitute words
    for j in range(len(original_clean_words)):
        gloss = original_gloss_words[j]
        if gloss in vocab and len(vocab[gloss]) > 0:

            new_word = np.random.choice(vocab[gloss])
            new_clean_words.append(new_word)
            new_gloss_words.append(gloss)
        else:
            new_clean_words.append(original_clean_words[j])
            new_gloss_words.append(gloss)

    
    new_clean_text = ' '.join(new_clean_words)
    new_gloss = ' '.join(new_gloss_words)
    
    new_original_text = new_clean_text
    
    new_sentences.append({
        'original_text': new_original_text,
        'clean_text': new_clean_text,
        'gloss': new_gloss,
    })

# --- 4. Create the new CSV ---
new_df = pd.DataFrame(new_sentences)
new_df.to_csv(f'generated_sentences_nash_random_{num_sentences_to_generate}.csv', index=False)

print("Generated 50 new sentences and saved to 'generated_sentences.csv'")
#print(vocab)
print("Sample of generated sentences:")
print(new_df.head())
print(len(new_df))
