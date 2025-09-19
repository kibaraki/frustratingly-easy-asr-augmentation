import pandas as pd

files = ['rand', 'gloss', 'gem']

for filename in files:
    df_samp = pd.read_csv(f"kakabe_{filename}.csv")
    df = df_samp.sample(n=1191) # to match rows in original train.csv  
    df.to_csv(f'{filename}-s-2.csv', index=False)

    csv = [f'{filename}-s-2.csv', 'train70.csv']

    all_dfs = []
    for c in csv:
        all_dfs.append(pd.read_csv(c))

    df = pd.concat(all_dfs, ignore_index=True)
    #df = df.drop_duplicates()

    #df.to_csv('kokoro/gpt+kf5+shib_inorder.csv', index=False)

    df = df.dropna()

    #df = pd.read_csv('gen_sent_kokoro-full_CLEANED.csv')
    df_shuf = df.sample(frac=1).reset_index(drop=True)
    df_shuf.to_csv(f'train70+{filename}-s-2.csv', index=False)


