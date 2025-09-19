import pandas as pd


csv1 = 'split/train70.csv'
df1 = pd.read_csv(csv1)
col1 = 'clean_text'

csv2 = 'split/train70.csv'
df2= pd.read_csv(csv2)
col2 = 'clean_text'


#full = pd.read_csv('split/combined_transcriptions-train-clean-100.csv')
#full_col = 'transcription'

# Extract unique words from the first column into a set
words1 = set(df1[col1].str.cat(sep=' ').split())
words1_list = df1[col1].apply(lambda x: len(x.split())).sum()
#print(f"size of train (all counts): {words1_list}")


# Extract unique words from the second column into a set
#words2 = set(df2[col2].str.cat(sep=' ').split())
#words2_list = df2[col2].apply(lambda x: len(x.split())).sum()


#full_words = set(full[full_col].str.cat(sep=' ').split())
#full_list = full[full_col].apply(lambda x: len(x.split())).sum()

# Return the number of common words
print(f"size of {csv1} (all counts): {words1_list}")
print(f"size of {csv1} (unique): {len(words1)}")
#print(f"size of {csv2} (all counts): {words2_list}")
#print(f"size of {csv2} (unique): {len(words2)}")

#print(f"intersection: {len(words1.intersection(words2))}")
#print(f"size of full (all counts): {full_list}")
#print(f"size of full (unique): {len(full_words)}")

