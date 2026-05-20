import pandas as pd

df = pd.read_csv('YARISMA_TRAIN_CFTR.csv')
df_temiz = df.dropna(axis=1, thresh=int(len(df) * 0.20))
df_tam_temiz = df_temiz.dropna(axis=0, thresh=int(df_temiz.shape[1] * 0.20))
df_tam_temiz.to_csv('TEMIZLENMIS_YARISMA_TRAIN.csv', index=False)