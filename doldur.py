import pandas as pd
from sklearn.impute import KNNImputer

# 1. Temizlenmiş veri setini okuyoruz
df = pd.read_csv('TEMIZLENMIS_YARISMA_TRAIN.csv')

# 2. Harfler ve Sayıları ayırıyoruz (KNN sadece sayılarla çalışır)
sayisal_veriler = df.select_dtypes(include=['float64', 'int64'])
metin_verileri = df.select_dtypes(exclude=['float64', 'int64'])

# 3. KNN Imputer'ı hazırlıyoruz (En benzer 5 komşuya bakacak)
imputer = KNNImputer(n_neighbors=5)

# 4. Sayısal verilerdeki boşlukları (NaN) komşulara bakarak dolduruyoruz
doldurulmus_sayilar = imputer.fit_transform(sayisal_veriler)

# 5. Doldurulan veriyi tekrar tablo formatına getiriyoruz
doldurulmus_sayilar_df = pd.DataFrame(doldurulmus_sayilar, columns=sayisal_veriler.columns)

# 6. Başta ayırdığımız metinleri ve doldurulmuş sayıları tekrar birleştiriyoruz
df_son = pd.concat([metin_verileri, doldurulmus_sayilar_df], axis=1)

# 7. Sonucu yeni bir dosya olarak kaydediyoruz
df_son.to_csv('DOLDURULMUS_YARISMA_TRAIN.csv', index=False)
print("İşlem Başarılı: Bütün boşluklar komşulara göre dolduruldu!")