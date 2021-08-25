import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import f1_score
import fasttext
import pdb
# 转换为FastText需要的格式
train_df = joblib.load(open('data/train_data.p','rb'))
train_df['label'] = [np.argmax(i) for i in train_df['label']]
train_df.to_csv('train_data_set.csv', index=None,header=None, sep='\t')
# pdb.set_trace()
train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
train_df[['text','label_ft']].iloc[:-10000].to_csv('train.csv', index=None, header=None, sep='\t')


model = fasttext.train_supervised('train.csv', lr=0.55, lrUpdateRate =100, wordNgrams=1,ws=5, dim=100,bucket=10000000,
                                  verbose=2, minCount=1, epoch=100, loss="hs")

val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-10000:]['text']]
print(f1_score(train_df['label'].values[-10000:].astype(str), val_pred, average='macro'))   # result: 19.2%
