# adjust tts generated csv into train.csv compatible

# vat
# sid,text,english,start_time,end_time,clean_text,segment_filename

# colab generated contains
# original_text,clean_text,gloss,original_text_no_paren,res_clip_filename


# kakabe train70.csv
# start_time,end_time,text,gloss,english_translation,clip_filename,clean_text


import pandas as pd

files = ['kakabe_gloss.csv', 'kakabe_gem.csv']

for filename in files:
    #filename = 'nash_gem.csv'
    print(filename)
    df = pd.read_csv(filename)

    df['english_translation'] = df['gloss']

    df['clip_filename'] = df['res_clip_filename']
    df['start_time'] = 10.0
    df['end_time'] = 11.0

    # original_text_no_prd_cmm,original_text_no_punc,original_text_mw,original_text_m,res_clip_filename,original_text_no_paren

    df['text'] = df['clean_text']

    df = df[['start_time', 'end_time', 'text', 'gloss', 'english_translation', 'clip_filename', 'clean_text']]


    df.to_csv(filename, index=False)



