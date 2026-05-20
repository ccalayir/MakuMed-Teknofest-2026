import pandas as pd

# 1. Yeni MASTER dosyasını okuyoruz
# low_memory=False komutu büyük dosyalarda uyarılardan kaçınmak içindir
df = pd.read_csv('YARISMA_TRAIN_MASTER.csv', low_memory=False)

# 2. %80'i boş olan SÜTUNLARI siliyoruz
min_dolu_sutun = int(len(df) * 0.20)
df_temiz = df.dropna(axis=1, thresh=min_dolu_sutun)

# 3. %80'i boş olan SATIRLARI siliyoruz
min_dolu_satir = int(df_temiz.shape[1] * 0.20)
df_tam_temiz = df_temiz.dropna(axis=0, thresh=min_dolu_satir)

# 4. Temizlenmiş halini klasöre kaydediyoruz
df_tam_temiz.to_csv('TEMIZLENMIS_MASTER.csv', index=False)
print("İşlem Başarılı! MASTER dosyası temizlendi ve kaydedildi.")