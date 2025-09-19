import pandas as pd

# sample to 10min, 30min, 60min, 90min
#df_samp = pd.read_csv('kokoro/shibur+bella+heart+fenrir+michael+emma.csv')
#df = df_samp.sample(frac=0.10) # for 1/10
#df.to_csv('kokoro/shibur+bella+heart+fenrir+michael+emma_0.1.csv', index=False)


# combine splits to finetune at same time


"""
csv = ['gpt_kokoro_af_bella.csv',
        'gpt_kokoro_af_heart.csv',
        'gpt_kokoro_am_fenrir.csv',
        'gpt_kokoro_am_michael.csv',
        'gpt_kokoro_bf_emma.csv'
]
"""

csv = ['343787.csv', '343790.csv', '343793.csv',  '343797.csv',  '343799.csv',  '343802.csv',  '343806.csv',  '343808.csv', '343788.csv' , '343791.csv',  '343794.csv',  '343798.csv',  '343801.csv' , '343803.csv',  '343807.csv']


#csv = ['shibur_train70_0_res_clean_kokoro_af_bella.csv', 'shibur_train70_0_res_clean_kokoro_af_heart.csv', 'shibur_train70_0_res_clean_kokoro_am_fenrir.csv', 'shibur_train70_0_res_clean_kokoro_am_michael.csv', 'shibur_train70_0_res_clean_kokoro_bf_emma.csv']

#csv= ['gem-hal.csv', 'train.csv']

#csv = ['kokoro/random_gem873.csv', 'kokoro/shibur+bella+heart+fenrir+michael+emma.csv', 'shibur_train70_0_res_clean.csv']

# csv = ['gen_sent_kokoro-full_CLEANED.csv', 'shibur_train70_0_res_clean.csv', 'kokoro/shibur+bella+heart+fenrir+michael+emma.csv']
"""
csv = ['shibur_train70_0_res_clean.csv', 
        #'shibur_train70_0_res_clean_kokoro_bf_emma.csv', 
        #'shibur_train70_0_res_clean_kokoro_am_michael.csv',
        #'shibur_train70_0_res_clean_kokoro_am_fenrir.csv',
        #'shibur_train70_0_res_clean_kokoro_af_heart.csv',
        'shibur_train70_0_res_clean_kokoro_af_bella.csv'
        ]
"""

"""
csv = ['shibur_train70_0_res_clean.csv', 
        'vc/don-conv_train70_0_res_clean.csv',
        'vc/don-conv_test15_0_res_clean.csv', 
        'vc/don-conv_val15_0_res_clean.csv', 
        'vc/dug-conv_train70_0_res_clean.csv',
        'vc/dug-conv_test15_0_res_clean.csv',
        'vc/dug-conv_val15_0_res_clean.csv',
        'vc/abi-conv_train70_0_res_clean.csv',
        'vc/abi-conv_test15_0_res_clean.csv', 
        'vc/abi-conv_val15_0_res_clean.csv', 
        'vc/jam-conv_train70_0_res_clean.csv',
        'vc/jam-conv_test15_0_res_clean.csv',
        'vc/jam-conv_val15_0_res_clean.csv']
"""

#['dug_train70_0_res_clean.csv', 'shibur_train70_0_res_clean.csv']#'not_dondog_train70_0_res_clean.csv'] 
#['shibur_train70_0_res_clean.csv', 'shibur_train70_0_res_clean.csv', 'shibur_train70_0_res_clean.csv'] #['kal_train70_20th_0.csv', 'monv_train70_12th_0.csv']  
# 'mon_kal/mon_train70_0.csv'
#  'shibur_train70_0_res.csv'


all_dfs = []
for c in csv:
    all_dfs.append(pd.read_csv(c))

df = pd.concat(all_dfs, ignore_index=True)
#df = df.drop_duplicates()

#df.to_csv('kokoro/gpt+kf5+shib_inorder.csv', index=False)


# mix up the rows

#df = pd.read_csv('gen_sent_kokoro-full_CLEANED.csv')
df_shuf = df.sample(frac=1).reset_index(drop=True)
df_shuf.to_csv('combined.csv', index=False)




# taking 1/5, 1/10, 1/20 of kal bc 48hrs
# taking 1/3, 1/6, 1/10, 1/12 of mon bc 29 hrs
"""
#csv = ['kal_train70_0.csv', 'kal_val15_0.csv']
csv = ['monv_train70_0.csv', 'monv_val15_0.csv']

for c in csv:
    df = pd.read_csv(c)
    fifth = len(df) // 12
    ind = c.index('_0')
    new_name = c[:ind] + '_12th_0.csv'
    df.iloc[:fifth].to_csv(new_name, index=False)
"""

