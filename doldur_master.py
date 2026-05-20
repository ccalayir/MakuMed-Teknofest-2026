import pandas as pd
from sklearn.impute import KNNImputer

print("Veri okunuyor, lütfen bekleyin...")
# 1. Temizlenmiş büyük dosyamızı okuyoruz
df = pd.read_csv('TEMIZLENMIS_MASTER.csv', low_memory=False)

# 2. Metinleri (T/T, C/C gibi genetik harfler) ve sayıları ayırıyoruz
sayisal_veriler = df.select_dtypes(include=['float64', 'int64'])
metin_verileri = df.select_dtypes(exclude=['float64', 'int64'])

# 3. KNN algoritmasını 5 komşuya bakacak şekilde hazırlıyoruz
print("Boşluklar yapay zeka ile dolduruluyor, bu işlem birkaç saniye sürebilir...")
imputer = KNNImputer(n_neighbors=5)
doldurulmus_sayilar = imputer.fit_transform(sayisal_veriler)

# 4. Tabloyu tekrar eski haline getiriyoruz
doldurulmus_sayilar_df = pd.DataFrame(doldurulmus_sayilar, columns=sayisal_veriler.columns)
df_son = pd.concat([metin_verileri, doldurulmus_sayilar_df], axis=1)

# 5. Tamamen doldurulmuş yeni dosyayı kaydediyoruz
df_son.to_csv('DOLDURULMUS_MASTER.csv', index=False)
print("İşlem Başarılı! Tüm eksik veriler dolduruldu ve DOLDURULMUS_MASTER.csv dosyası kaydedildi.")